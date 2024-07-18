use log::{Level, LevelFilter, Metadata, Record};
use std::sync::Once;

static INIT: Once = Once::new();

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!(
                "{} - {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

pub fn init() {
    INIT.call_once(|| {
        log::set_logger(&SimpleLogger).unwrap();
        log::set_max_level(LevelFilter::Info);
    });
}

pub fn set_log_level(level: LevelFilter) {
    log::set_max_level(level);
}
