
use wire::*;
use wire::{IntoReactive, React};

use std::io::{self, Write};
use std::iter::{repeat};


fn main() {
    let clock = Clock{step_type: StepType::Continuous, target_fps : 10.0};


    let anim = wloop(fors(0.5, "Hoo...").then("...ray!".fors(0.5)));
    let mut max_len =0;

    clock.run(
        wloop((fors(2,"Once upon a time...")
            .then("... games were completely imperative ...".fors(3))
            .then("... but then ...".fors(2)).wmap(|x| x.fmap(String::from)))
              .then((String::from("Wires ! ").into_reactive() + anim).fors(10))),
        |s| {
            let string_len = s.len();
            let spaces =  if string_len < max_len {
                max_len - string_len
            } else {
                0
            };
            let line : String = s.chars().chain(repeat(' ').take(spaces)).collect();
            print!("{}\r",line);
            io::stdout().flush().unwrap();
            max_len = max_len.max(string_len);
            true
        }
    )



}
