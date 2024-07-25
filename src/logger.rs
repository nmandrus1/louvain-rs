use env_logger::Builder;
use log::LevelFilter;
use std::io::Write;
use std::sync::Once;
use std::sync::OnceLock;

static INIT: Once = Once::new();

static RANK: OnceLock<usize> = OnceLock::new();

pub fn init(rank: usize) {
    INIT.call_once(|| {
        RANK.set(rank).unwrap();

        Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format(|buf, record| {
                let rank = RANK.get().expect("lgger.rs: RANK not initialized");
                writeln!(
                    buf,
                    "[RANK {}] {} - {}: {}",
                    rank,
                    record.level(),
                    record.target(),
                    record.args()
                )
            })
            .init();
    });
}

pub fn set_log_level(level: LevelFilter) {
    log::set_max_level(level);
}
