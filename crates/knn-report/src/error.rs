//! Why a report could not be built.

/// Why a report could not be built. Each step that failed names itself
/// and wraps why, down to the format, search, I/O, or JSON error at the
/// bottom, so the chain of sources read in order is the command's error
/// line: `opening x.faiss: The system cannot find the file specified.`
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A step that failed, and why.
    #[error("{what}")]
    In {
        /// The step, with the path it was taken on.
        what: String,
        /// Why it failed.
        #[source]
        why: Box<Error>,
    },
    /// A file did not open or parse.
    #[error(transparent)]
    Format(#[from] knn_format::Error),
    /// A search or a score could not be run.
    #[error(transparent)]
    Recall(#[from] knn_recall::Error),
    /// A query, result, or id file could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A result or id file is not the JSON expected.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// What is wrong, in words: the file, the options, and the inputs
    /// disagree.
    #[error("{0}")]
    Message(String),
    /// Options that contradict each other, or lie outside the ranges
    /// k-NN accepts; the text names the first at fault by its flag.
    #[error("{0}")]
    Options(String),
    /// A failure after a warning: it reads as the failure, and
    /// [`Error::warning`] gives the warning, to be said first.
    #[error(transparent)]
    Warned(Box<Warned>),
}

impl Error {
    /// What was warned of before the failure, if anything: for a graph
    /// without its vectors, what scan's check of the field's files
    /// warns of.
    #[must_use]
    pub fn warning(&self) -> Option<&str> {
        match self {
            Self::Warned(w) => Some(&w.warning),
            Self::In { why, .. } => why.warning(),
            _ => None,
        }
    }
}

/// A failure that came after a warning. It displays as the failure,
/// and its sources are the failure's.
#[derive(Debug)]
pub struct Warned {
    /// What was warned of.
    pub warning: String,
    /// The failure.
    pub error: Error,
}

impl std::fmt::Display for Warned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for Warned {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.error)
    }
}

/// A report, or why it could not be built.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Names the step a failure happened in, as `anyhow::Context` does.
pub(crate) trait Context<T> {
    /// Wraps the error, or a `None`, in the step `what` names.
    fn with_context(self, what: impl FnOnce() -> String) -> Result<T>;
}

impl<T, E: Into<Error>> Context<T> for Result<T, E> {
    fn with_context(self, what: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|e| Error::In {
            what: what(),
            why: Box::new(e.into()),
        })
    }
}

impl<T> Context<T> for Option<T> {
    fn with_context(self, what: impl FnOnce() -> String) -> Result<T> {
        self.ok_or_else(|| Error::Message(what()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The command prints the chain through `anyhow`'s alternate form,
    /// which walks the sources: each link's text, joined by `: `.
    fn line(e: &Error) -> String {
        let mut s = e.to_string();
        let mut source = std::error::Error::source(e);
        while let Some(inner) = source {
            s = format!("{s}: {inner}");
            source = inner.source();
        }
        s
    }

    #[test]
    fn the_chain_reads_as_the_error_line() {
        let io = std::io::Error::other("no such file");
        let e = Err::<(), _>(io)
            .with_context(|| "reading q.f32".to_owned())
            .with_context(|| "x.faiss: queries".to_owned())
            .unwrap_err();
        assert_eq!(line(&e), "x.faiss: queries: reading q.f32: no such file");
        let none = None::<u8>.with_context(|| "no field 3".to_owned());
        assert_eq!(line(&none.unwrap_err()), "no field 3");
        assert_eq!(
            line(&Error::Message("give --queries".to_owned())),
            "give --queries"
        );
    }

    #[test]
    fn a_warned_failure_reads_as_the_failure() {
        let io = std::io::Error::other("no such file");
        let failure = Err::<(), _>(io)
            .with_context(|| "reading q.f32".to_owned())
            .unwrap_err();
        let e = Error::Warned(Box::new(Warned {
            warning: "1 record differs".to_owned(),
            error: failure,
        }));
        assert_eq!(line(&e), "reading q.f32: no such file");
        assert_eq!(e.warning(), Some("1 record differs"));
        // Found under a step that wraps it, too.
        let wrapped = Err::<(), _>(e)
            .with_context(|| "x.faiss".to_owned())
            .unwrap_err();
        assert_eq!(wrapped.warning(), Some("1 record differs"));
        assert_eq!(line(&wrapped), "x.faiss: reading q.f32: no such file");
        assert_eq!(Error::Options("--k".to_owned()).warning(), None);
    }
}
