use std::string::String;

error_chain! {
  errors {
    TryFromTVMRetValueError(expected: String, actual: i64) {
      description("mismatched types while downcasting TVMRetValue")
      display("invalid downcast: expected `{}` but was `{}`", expected, actual)
    }
  }
}
