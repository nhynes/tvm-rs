use std::alloc::{self, Layout};

use errors::*;

const DEFAULT_ALIGN_BYTES: usize = 4;

#[derive(PartialEq, Eq)]
pub struct Allocation {
  layout: Layout,
  ptr: *mut u8,
}

impl Allocation {
  pub fn new(size: usize, align: Option<usize>) -> Result<Self> {
    let alignment = align.unwrap_or(DEFAULT_ALIGN_BYTES);
    let layout = Layout::from_size_align(size, alignment)?;
    let ptr = unsafe { alloc::alloc(layout.clone()) };
    if ptr.is_null() {
      alloc::oom(layout);
    }
    Ok(Self {
      ptr: ptr,
      layout: layout,
    })
  }

  pub fn as_mut_ptr(&self) -> *mut u8 {
    self.ptr
  }

  pub fn size(&self) -> usize {
    self.layout.size()
  }
}

impl Drop for Allocation {
  fn drop(&mut self) {
    unsafe {
      alloc::dealloc(self.ptr, self.layout.clone());
    }
  }
}
