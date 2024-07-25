// use super::CommunicationLayer;
// use mpi::topology::{Communicator, SimpleCommunicator};
// use mpi::traits::{CommunicatorCollectives, Equivalence};

// pub struct MPICommunicationLayer<'mpi> {
//     communicator: &'mpi SimpleCommunicator,
// }

// impl<'mpi> MPICommunicationLayer<'mpi> {
//     pub fn new(communicator: &SimpleCommunicator) -> Self {
//         Self { communicator }
//     }
// }

// impl<'mpi> CommunicationLayer for MPICommunicationLayer<'mpi> {
//     /// send
//     fn send<T: Equivalence>(&self, data: &T, dest: mpi::Rank) -> anyhow::Result<()> {}

//     fn receive<T: Equivalence>(&self, source: mpi::Rank) -> anyhow::Result<T> {
//         todo!()
//     }

//     fn broadcast<T: Equivalence>(&self, data: &mut T, root: mpi::Rank) -> anyhow::Result<()> {
//         todo!()
//     }

//     fn all_reduce<T: Equivalence>(&self, data: &T, op: crate::AllReduceOp) -> anyhow::Result<T> {
//         todo!()
//     }

//     fn barrier(&self) -> anyhow::Result<()> {
//         todo!()
//     }

//     fn rank(&self) -> mpi::Rank {
//         todo!()
//     }

//     fn size(&self) -> usize {
//         todo!()
//     }
// }
