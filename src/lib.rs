mod louvain;
pub use louvain::*;
use mpi::{environment::Universe, topology::SimpleCommunicator, traits::Communicator};

mod logger;
pub use logger::set_log_level;

/// Initializes MPI and Logging
pub fn init() -> anyhow::Result<(Universe, SimpleCommunicator)> {
    let universe = mpi::initialize().ok_or(anyhow::anyhow!("MPI Not Initialized"))?;
    let world = universe.world();

    logger::init(world.rank() as usize);

    Ok((universe, world))
}
