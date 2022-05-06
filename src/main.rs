
use wire::*;

use std::io::{self, Write};


fn main() {
    let clock = Clock{step_type: StepType::Continuous, target_fps : 60.0};


    // let anim = wloop(fors(0.5, "Hoo...").then("...ray!".fors(0.5)));

    // clock.run(
    //     wloop(fors(2,"Once upon a time...")
    //         .then("... games were completely imperative ...".fors(3))
    //         .then("... but then ...".fors(2))
    //         .then((String::from("Wires ! ") + anim).fors(10)))
    //     |s| {
    //         print!("{}\r",s);
    //         io::stdout().flush().unwrap();
    //     }
    // )



}
