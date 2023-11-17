pub struct PerfCounter {
    #[cfg(feature = "print-perf")]
    name: &'static str,
    #[cfg(feature = "print-perf")]
    start: std::time::Instant,
}

impl PerfCounter {
    #[cfg(not(feature = "print-perf"))]
    pub fn new(_name: &'static str) -> Self {
        PerfCounter {}
    }

    #[cfg(feature = "print-perf")]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for PerfCounter {
    fn drop(&mut self) {
        #[cfg(feature = "print-perf")]
        {
            let elapsed = self.start.elapsed();
            eprintln!("[v8-heap-perf] {}: {}ms", self.name, elapsed.as_millis());
        }
    }
}
