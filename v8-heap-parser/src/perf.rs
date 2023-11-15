pub struct PerfCounter {
    #[cfg(feature = "webp")]
    name: &'static str,
    #[cfg(feature = "webp")]
    start: std::time::Instant,
}

impl PerfCounter {
    #[cfg(not(feature = "webp"))]
    pub fn new(_name: &'static str) -> Self {
        PerfCounter {}
    }

    #[cfg(feature = "webp")]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for PerfCounter {
    fn drop(&mut self) {
        #[cfg(feature = "webp")]
        {
            let elapsed = self.start.elapsed();
            eprintln!("[v8-heap-perf] {}: {}ms", self.name, elapsed.as_millis());
        }
    }
}
