use distributed_louvain::*;

use anyhow;
use mpi::topology::Communicator;
use mpi::traits::*;

fn run_mpi_tests() -> anyhow::Result<()> {
    // Your test functions go here
    test_vertex_distribution()?;
    test_from_distributed()?;
    // Add more test functions as needed
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    run_mpi_tests()?;

    // Ensure all processes are done before finalizing MPI
    world.barrier();
    Ok(())
}

// Test functions

fn test_vertex_distribution() -> anyhow::Result<()> {
    let world = mpi::initialize()
        .ok_or(anyhow::anyhow!("Failed to initialize MPI"))?
        .world();

    let rank = world.rank();
    let size = world.size();

    // Test DistributedInfo::compute_vertex_distribution
    let global_vcount = 100;
    let distribution =
        DistributedInfo::<usize>::compute_vertex_distribution(size as usize, global_vcount);

    // Check that the distribution is correct
    assert_eq!(distribution.len(), size as usize);
    assert_eq!(
        distribution
            .iter()
            .map(|(start, end)| end - start)
            .sum::<usize>(),
        global_vcount
    );

    // Check that each rank's portion is correct
    let (start, end) = distribution[rank as usize];
    let expected_count =
        global_vcount / size as usize + if rank < global_vcount % size { 1 } else { 0 };
    assert_eq!(end - start, expected_count);

    if rank == 0 {
        println!("test_vertex_distribution passed");
    }
    Ok(())
}

fn test_from_distributed() -> anyhow::Result<()> {
    let world = mpi::UNIVERSE.world();
    let rank = world.rank();
    let size = world.size();

    // Create some test edges
    let edges = if rank == 0 {
        vec![(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)]
    } else {
        vec![]
    };

    // Call from_distributed
    let graph = DistributedGraph::<usize, (), f32>::from_distributed(&edges).unwrap();

    // Check that the graph was created correctly
    assert_eq!(graph.global_vcount, 4);

    // Check local properties
    let expected_local_vcount = 4 / size as usize + if rank < 4 % size { 1 } else { 0 };
    assert_eq!(graph.local_vcount, expected_local_vcount);

    // You might want to add more specific checks here depending on how your graph is structured

    if rank == 0 {
        println!("test_from_distributed passed");
    }
    Ok(())
}
