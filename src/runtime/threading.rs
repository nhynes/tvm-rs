use std::{
  env,
  os::raw::c_void,
  sync::{
    atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT},
    Arc,
    Barrier,
  },
  thread::{self, JoinHandle},
};

#[cfg(not(target_env = "sgx"))]
use num_cpus;

use bounded_spsc_queue::{self, Consumer, Producer};

use super::super::errors::*;
use ffi::runtime::TVMParallelGroupEnv;

type FTVMParallelLambda =
  extern "C" fn(task_id: usize, penv: *const TVMParallelGroupEnv, cdata: *const c_void) -> i32;

pub struct ThreadPool {
  num_workers: usize,
  queues: Vec<Producer<Task>>,
  #[allow(unused)]
  threads: Vec<JoinHandle<()>>,
}

// thread_local!(static THREAD_POOL: RefCell<ThreadPool> = RefCell::new(ThreadPool::new()));
thread_local!(static THREAD_POOL: ThreadPool = ThreadPool::new());

pub struct Job {
  cb: FTVMParallelLambda,
  cdata: *const c_void,
  req_num_tasks: usize,
  pending: Arc<AtomicUsize>,
}

impl Job {
  fn tasks(&self, num_workers: usize) -> Vec<Task> {
    let num_tasks = if self.req_num_tasks == 0 {
      num_workers
    } else {
      self.req_num_tasks.min(num_workers)
    };
    self.pending.store(num_tasks, Ordering::SeqCst);

    let barrier = Arc::new(Barrier::new(num_tasks));

    (0..num_tasks)
      .map(move |i| Task {
        id: i,
        flambda: self.cb,
        penv: TVMParallelGroupEnv {
          sync_handle: &Arc::clone(&barrier) as *const _ as *mut c_void,
          num_task: num_tasks as i32,
        },
        cdata: self.cdata,
        pending: Arc::clone(&self.pending),
      })
      .collect()
  }

  fn wait(&self) -> Result<()> {
    while self.pending.load(Ordering::Acquire) > 0 {
      thread::yield_now();
    }
    Ok(())
  }
}

struct Task {
  id: usize,
  flambda: FTVMParallelLambda,
  penv: TVMParallelGroupEnv,
  cdata: *const c_void,
  pending: Arc<AtomicUsize>,
}
unsafe impl Send for Task {}

impl ThreadPool {
  #[cfg(not(target_env = "sgx"))]
  fn new() -> Self {
    let num_workers = max_concurrency();

    let mut producers = Vec::new();
    let mut threads = Vec::new();
    for i in 0..num_workers {
      let (p, c) = bounded_spsc_queue::make(2);
      producers.push(p);
      threads.push(thread::spawn(move || ThreadPool::run_worker(c, i)));
    }

    ThreadPool {
      num_workers: num_workers,
      queues: producers,
      threads: threads,
    }
  }

  #[cfg(target_env = "sgx")]
  fn new() -> Self {
    let num_workers = max_concurrency();

    // tvm_ocall_request_workers(num_workers);

    ThreadPool {
      num_workers: num_workers,
      queues: Vec::new(),
      threads: Vec::new(),
    }
  }

  fn launch(&self, job: Job) {
    let tasks = job.tasks(self.num_workers);

    let _: Vec<()> = tasks
      .into_iter()
      .zip(self.queues.iter())
      .map(|(task, q)| q.push(task))
      .collect();

    job.wait().unwrap();
  }

  fn run_worker(q: Consumer<Task>, worker_id: usize) {
    loop {
      let task = q.pop();
      let status = (task.flambda)(task.id, &task.penv as *const _, task.cdata);
      task.pending.fetch_sub(1, Ordering::AcqRel);
      if status != 0 {
        panic!(format!("Error in task {}", worker_id));
      }
    }
  }
}

#[cfg(not(target_env = "sgx"))]
fn max_concurrency() -> usize {
  if let Ok(threads_str) = env::var("TVM_NUM_THREADS").or(env::var("OMP_NUM_THREADS")) {
    if let Ok(threads) = usize::from_str_radix(&threads_str, 10) {
      return threads;
    }
  }
  num_cpus::get_physical()
}

#[cfg(target_env = "sgx")]
fn max_concurrency() -> usize {
  usize::from_str_radix(env!("TVM_NUM_THREADS"), 10).unwrap_or(1)
}

#[no_mangle]
pub extern "C" fn TVMBackendParallelLaunch(
  cb: FTVMParallelLambda,
  cdata: *const c_void,
  num_task: usize,
) {
  THREAD_POOL.with(|pool| {
    pool.launch(Job {
      cb: cb,
      cdata: cdata,
      req_num_tasks: num_task,
      pending: Arc::new(ATOMIC_USIZE_INIT),
    });
  });
}

#[no_mangle]
pub extern "C" fn TVMBackendParallelBarrier(_task_id: usize, penv: *const TVMParallelGroupEnv) {
  let barrier: &Arc<Barrier> = unsafe { &*((*penv).sync_handle as *const Arc<Barrier>) };
  barrier.wait();
}

#[cfg(test)]
mod tests {
  use std::{ptr, thread, time::Duration};

  use super::*;

  #[test]
  fn test_max_concurrency() {
    env::set_var("TVM_NUM_THREADS", "42");
    env::set_var("OMP_NUM_THREADS", "24");
    assert_eq!(max_concurrency(), 42);
    env::remove_var("TVM_NUM_THREADS");
    assert_eq!(max_concurrency(), 24);
  }

  extern "C" fn flambda(
    task_id: usize,
    penv: *const TVMParallelGroupEnv,
    cdata: *const c_void,
  ) -> i32 {
    if cdata == ptr::null() {
      return 0;
    }
    unsafe {
      let &(ref counter, ref task_ids_sum) = &*(cdata as *const (AtomicUsize, AtomicUsize));
      thread::sleep(Duration::from_millis(50 * task_id as u64));
      counter.fetch_add(1, Ordering::SeqCst);
      task_ids_sum.fetch_add(task_id, Ordering::SeqCst);
      assert_eq!((*penv).num_task, 3);
    }
    0
  }

  #[test]
  fn test_parallel_launch() {
    TVMBackendParallelLaunch(flambda, ptr::null(), 6);
    let counter = ATOMIC_USIZE_INIT;
    let task_ids_sum = ATOMIC_USIZE_INIT;
    let cdata = (counter, task_ids_sum);
    let num_tasks = 3;
    TVMBackendParallelLaunch(flambda, &cdata as *const _ as *const c_void, num_tasks);
    assert_eq!(cdata.0.load(Ordering::SeqCst), num_tasks);
    assert_eq!(cdata.1.load(Ordering::SeqCst), (0..num_tasks).sum());
  }
}
