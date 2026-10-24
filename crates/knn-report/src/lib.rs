//! The reports behind `knn-inspect`'s commands: `info`, `verify`, `stats`,
//! `recall`, and `scan`. Each module builds its command's report from the
//! library crates (`knn-format`, `knn-check`, `knn-recall`) and returns it
//! as a value that serializes to the command's JSON, with a `text`
//! function that renders it as the command prints it. The binary parses
//! the arguments and prints; another front end builds the same reports.
//!
//! An [`Error`] names the step that failed and wraps why; its chain of
//! sources, read in order, is the command's error line. `recall` refuses
//! what the command line refuses: options that contradict each other or
//! lie outside the ranges k-NN accepts. Errors name an option by its
//! command-line flag, `--nprobes` for `recall::Options::nprobes`, so the
//! library and the command say the same.

/// An [`Error::Message`] from a format string.
macro_rules! message {
    ($($arg:tt)*) => {
        $crate::Error::Message(format!($($arg)*))
    };
}

/// Returns an [`Error::Message`] from a format string.
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err(message!($($arg)*))
    };
}

/// `writeln!` into a `String`, which cannot fail.
macro_rules! push_line {
    ($s:expr, $($arg:tt)*) => {
        ::std::writeln!($s, $($arg)*).expect("writing to a String cannot fail")
    };
}

mod error;
pub mod info;
pub mod recall;
pub mod scan;
pub mod stats;
pub mod verify;

pub use error::{Error, Result, Warned};
