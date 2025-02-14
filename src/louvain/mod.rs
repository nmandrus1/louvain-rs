mod communities;
mod communitystate;
mod graph;
mod messagemanager;
mod traits;

use messagemanager::*;
pub use traits::*;

pub use communities::*;
pub use communitystate::*;
pub use graph::*;

use mpi::Count;

// Helper functions

/// Given a list of counts, create a list of displacements
fn displs_from_counts(counts: &[Count]) -> Vec<Count> {
    counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect()
}
