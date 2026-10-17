//! Structural checks and graph statistics over parsed k-NN segment files.
//!
//! [`verify`] runs a fixed list of [`Check`]s and returns a [`Report`]:
//! one [`CheckResult`] per check, each with its worst [`Severity`] and
//! the first few [`Finding`]s, every finding carrying the file offset
//! and node it points at. Nothing here panics on malformed input: a file
//! that does not parse is reported as a failed [`Check::Structure`].
//!
//! [`graph_stats`] reports per-level node counts, out-degree histograms,
//! and the nodes no search can reach.

mod checks;
mod stats;

use serde::Serialize;

pub use checks::verify;
pub use stats::{GraphStats, LayerStats, graph_stats};

/// How much a finding matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Worth knowing; the file is fine.
    Info,
    /// The file is usable but something about it is off.
    Warning,
    /// The file is corrupt or not what it claims to be.
    Failure,
}

/// The space type a field was created with, for checks the file itself
/// cannot know about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Space {
    /// `l2`.
    #[serde(rename = "l2")]
    L2,
    /// `innerproduct`.
    #[serde(rename = "innerproduct")]
    InnerProduct,
    /// `cosinesimil`: stored as inner product over unit vectors.
    #[serde(rename = "cosinesimil")]
    Cosine,
    /// `hamming`: the only space k-NN accepts for binary fields.
    #[serde(rename = "hamming")]
    Hamming,
}

impl Space {
    /// The space's name in k-NN's spelling, as it also serializes.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::InnerProduct => "innerproduct",
            Self::Cosine => "cosinesimil",
            Self::Hamming => "hamming",
        }
    }
}

/// What [`verify`] should know beyond the bytes.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The field's space type, if known. Required for the cosine checks;
    /// otherwise it only adds a metric mismatch check.
    pub space: Option<Space>,
    /// How many findings to keep per check (the count is always exact);
    /// 0 means [`Options::DEFAULT_MAX_EXAMPLES`].
    pub max_examples: usize,
}

impl Options {
    /// Findings kept per check when the caller does not say.
    pub const DEFAULT_MAX_EXAMPLES: usize = 5;
}

/// The checks, in the order they run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Check {
    /// The Lucene envelope: whether a footer is present (a bare Faiss
    /// index is accepted), and a warning if a header is.
    Envelope,
    /// The footer's CRC32 over the file.
    Checksum,
    /// The Faiss layout parses to the end.
    Structure,
    /// No bytes between the index and the footer.
    Trailing,
    /// The three Faiss headers agree and hold supported values; for
    /// scalar-quantized storage, the quantizer's dimension and code size
    /// match them and it holds no trained values; for binary storage, the
    /// dimension is a multiple of 8 and the code size is its bytes, and a
    /// metric type other than the one k-NN writes is a warning.
    Headers,
    /// `--space` matches the metric the file records. With `--space
    /// cosine`, byte storage is also reported, as a warning or, when the
    /// metric does not match either, in the failure: k-NN refuses
    /// cosinesimil for byte fields. Binary storage, which computes Hamming
    /// distance, passes as `hamming`, and with any other space is noted:
    /// k-NN writes it for float fields with the binary encoder too.
    Metric,
    /// Neighbor offsets are monotonic and end at the neighbor array.
    Offsets,
    /// Node levels match the per-node slot counts and the level table.
    Levels,
    /// The entry point exists and sits on the top level.
    EntryPoint,
    /// Every neighbor slot is -1 or a node id.
    NeighborIds,
    /// Every neighbor listed on a level is itself on that level. A search
    /// that steps to one that is not reads the wrong node's slots.
    NeighborLevels,
    /// The storage holds exactly `count x dimension` components: float32
    /// elements, two bytes each of fp16 or bf16 codes, or one byte each of
    /// byte codes; binary storage, `count x code size` bytes.
    VectorCount,
    /// The ID map holds one label per node.
    IdMapLength,
    /// No NaN or infinite vector components. Skipped for byte and binary
    /// storage, where no code is either.
    VectorValues,
    /// No node lists the same neighbor twice on one level.
    DuplicateNeighbors,
    /// No node lists itself as a neighbor.
    SelfLoops,
    /// With `--space cosine`: every vector has unit length. Skipped for
    /// byte storage, which k-NN never normalizes, since it refuses
    /// cosinesimil for byte fields (the metric check reports the space
    /// instead), and for binary storage, whose bits have no length.
    Norms,
}

impl Check {
    /// Every check, in run order.
    pub const ALL: [Self; 17] = [
        Self::Envelope,
        Self::Checksum,
        Self::Structure,
        Self::Trailing,
        Self::Headers,
        Self::Metric,
        Self::Offsets,
        Self::Levels,
        Self::EntryPoint,
        Self::NeighborIds,
        Self::NeighborLevels,
        Self::VectorCount,
        Self::IdMapLength,
        Self::VectorValues,
        Self::DuplicateNeighbors,
        Self::SelfLoops,
        Self::Norms,
    ];

    /// The check's name as it appears in output and manifests.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Envelope => "envelope",
            Self::Checksum => "checksum",
            Self::Structure => "structure",
            Self::Trailing => "trailing",
            Self::Headers => "headers",
            Self::Metric => "metric",
            Self::Offsets => "offsets",
            Self::Levels => "levels",
            Self::EntryPoint => "entry_point",
            Self::NeighborIds => "neighbor_ids",
            Self::NeighborLevels => "neighbor_levels",
            Self::VectorCount => "vector_count",
            Self::IdMapLength => "id_map_length",
            Self::VectorValues => "vector_values",
            Self::DuplicateNeighbors => "duplicate_neighbors",
            Self::SelfLoops => "self_loops",
            Self::Norms => "norms",
        }
    }

    /// The check with that [`name`](Self::name).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }
}

/// One thing a check found.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Finding {
    /// What was found.
    pub message: String,
    /// Where in the file, when the finding points at bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    /// Which node, when the finding is about one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<usize>,
}

/// The outcome of one check.
#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    /// The check; in JSON it appears once, as `name`.
    #[serde(skip)]
    pub check: Check,
    /// The check's name.
    pub name: &'static str,
    /// The worst severity among its findings; `None` when it passed
    /// cleanly or was skipped.
    pub severity: Option<Severity>,
    /// Why the check did not run, when it did not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
    /// How many findings there were in total.
    pub count: usize,
    /// The first findings, up to [`Options::max_examples`].
    pub findings: Vec<Finding>,
}

impl CheckResult {
    /// Whether the check reported a failure.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.severity == Some(Severity::Failure)
    }
}

/// The overall verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// No failures, no warnings.
    Clean,
    /// Warnings only.
    Warnings,
    /// At least one failure.
    Failed,
}

impl Status {
    /// The process exit code the plan assigns: 0 clean, 1 failed, 2
    /// warnings only.
    #[must_use]
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Clean => 0,
            Self::Failed => 1,
            Self::Warnings => 2,
        }
    }
}

/// Everything [`verify`] found.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// The verdict.
    pub status: Status,
    /// Checks that reported a failure.
    pub failures: usize,
    /// Checks that reported a warning and nothing worse.
    pub warnings: usize,
    /// One entry per check, in run order.
    pub checks: Vec<CheckResult>,
}

impl Report {
    /// The result for one check.
    #[must_use]
    pub fn result(&self, check: Check) -> Option<&CheckResult> {
        self.checks.iter().find(|r| r.check == check)
    }
}
