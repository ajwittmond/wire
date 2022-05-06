#![allow(dead_code)]
extern crate derive_wire;

use derive_wire::react_traits;
use std::ops::{Add, BitAnd, BitOr, BitXor, Deref, Div, Index, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use std::time::{Duration, Instant};
pub trait Startable{
    /// establish a time offset for wires that act relative to a start time
    fn start(&mut self, time: Duration){}

}

pub trait React<Input>: Sized + Startable {
    type Output;
    fn react(&mut self, input: Input, time: Duration) -> Self::Output;

    fn after(self) {}

    fn alternate(self) {}

    fn at(self) {}

    fn fors(self) {}

    fn to<R>(self, other: R) -> Composed<Self, R>
    where
        R: React<Self::Output>,
    {
        Composed(self, other)
    }
}

pub trait EventAndValueEmmiting<I>: Sized {
    type EventType;
    type ValueType;

    fn then<R>(self, next: R) -> Then<Self, R>
    where
        R: React<I, Output = Self::ValueType>,
    {
        Then {
            wire1: self,
            wire2: next,
            switched: false,
        }
    }

    fn wloop(self) {}
}

pub trait EventEmmiting {}

fn at() {}

fn wloop() {}

fn fors() {}

// impl<I,E,C : Copy> React<I,E> for C {
//     type Output = C;

//     fn react(&mut self, input: I, time: E) -> Self::Output{
//         *self
//     }
// }
macro_rules! copy_react_impls{
    ($($type:ty),*) => {
        $(
            impl Startable for $type {} 
            impl<I> React<I> for $type{
                type Output = Self;

                fn react(&mut self, _input : I, _time: Duration) -> Self{
                    self.clone()
                }

            }
        ) *
    }
}

copy_react_impls![String, &str, char, i8, i16, i32, i64, f32, f64, u8, u16, u32, u64, bool];

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct EventValue<E> {
    pub value: E,
    pub time: Duration,
}
impl<E> EventValue<E> {
    fn map<E2, F: FnOnce(E) -> E2>(self, f: F) -> EventValue<E2> {
        EventValue {
            value: f(self.value),
            time: self.time,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Event<E> {
    Event(EventValue<E>),
    NoEvent,
}

impl<E> Event<E> {
    fn map<E2, F: FnOnce(E) -> E2>(self, f: F) -> Event<E2> {
        use crate::Event::*;
        match self {
            Event(e) => Event(e.map(f)),
            NoEvent => NoEvent,
        }
    }

    fn merge(self, other: Event<E>) -> Event<E> {
        use crate::Event::*;
        match self {
            NoEvent => other,
            Event(e) => Event(e),
        }
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Func<F>(F);
impl<F> Startable for Func<F> {}
impl<I, O, F: Fn(I, Duration) -> O> React<I> for Func<F> {
    type Output = O;

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        self.0(input, time)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct StatefulFunc<F, S>(F, S);
impl<F,S> Startable for StatefulFunc<F,S> {}
impl<I, O, S, F: Fn((I, &mut S), Duration) -> O> React<I> for StatefulFunc<F, S> {
    type Output = O;

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        self.0((input, &mut self.1), time)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct First<F>(F);

impl<F : Startable> Startable for First<F>{
    fn start(&mut self, start: Duration){
        self.0.start(start)
    }
}
impl<I, T, F: React<I>> React<(I, T)> for First<F> {
    type Output = (F::Output, T);

    fn react(&mut self, input: (I, T), time: Duration) -> Self::Output {
        (self.0.react(input.0, time), input.1)
    }

    
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Second<F>(F);

impl<F:Startable> Startable for Second<F>{
    fn start(&mut self, start: Duration){
        self.0.start(start)
    }
}
impl<I, T, F: React<I>> React<(T, I)> for Second<F> {
    type Output = (T, F::Output);

    fn react(&mut self, input: (T, I), time: Duration) -> Self::Output {
        (input.0, self.0.react(input.1, time))
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Composed<F1, F2>(F1, F2);

impl<F1:Startable,F2:Startable> Startable for Composed<F1,F2>{
    fn start(&mut self, start: Duration){
        self.0.start(start);
        self.1.start(start);
    }
}
impl<I, F1: React<F2::Output>, F2: React<I>> React<I> for Composed<F1, F2> {
    type Output = F1::Output;

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        self.0.react(self.1.react(input, time), time)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct FanOut<F1, F2>(F1, F2);
impl<F1:Startable,F2:Startable> Startable for FanOut<F1,F2>{
    fn start(&mut self, start: Duration){
        self.0.start(start);
        self.1.start(start);
    }
}
impl<I1, I2, F1: React<I1>, F2: React<I2>> React<(I1, I2)> for FanOut<F1, F2> {
    type Output = (F1::Output, F2::Output);

    fn react(&mut self, (i1, i2): (I1, I2), time: Duration) -> Self::Output {
        (self.0.react(i1, time), self.1.react(i2, time))
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Split<F1, F2>(F1, F2);
impl<F1:Startable,F2:Startable> Startable for Split<F1,F2>{
    fn start(&mut self, start: Duration){
        self.0.start(start);
        self.1.start(start);
    }
}
impl<I: Clone, F1: React<I>, F2: React<I>> React<I> for Split<F1, F2> {
    type Output = (F1::Output, F2::Output);

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        (
            self.0.react(input.clone(), time),
            self.1.react(input, time),
        )
    }

}

pub struct Pure<C>(C);
impl<C> Startable for Pure<C>{}
impl<I, C: Clone> React<I> for Pure<C> {
    type Output = C;

    fn react(&mut self, _input: I, _time: Duration) -> C {
        self.0.clone()
    }
}

trait Dt {
    fn dt(self, delta: Duration) -> Self;
    fn idt(self, delta: Duration) -> Self;
}

impl Dt for f32 {
    fn dt(self, delta: Duration) -> Self {
        self * delta.as_secs_f32()
    }
    fn idt(self, delta: Duration) -> Self {
        self / delta.as_secs_f32()
    }
}
impl Dt for f64 {
    fn dt(self, delta: Duration) -> Self {
        self * delta.as_secs_f64()
    }
    fn idt(self, delta: Duration) -> Self {
        self / delta.as_secs_f64()
    }
}
impl Dt for i32 {
    fn dt(self, delta: Duration) -> Self {
        self * (delta.as_secs() as i32)
    }
    fn idt(self, delta: Duration) -> Self {
        self / (delta.as_secs() as i32)
    }
}
impl Dt for u32 {
    fn dt(self, delta: Duration) -> Self {
        self * (delta.as_secs() as u32)
    }
    fn idt(self, delta: Duration) -> Self {
        self / (delta.as_secs() as u32)
    }
}
impl Dt for i64 {
    fn dt(self, delta: Duration) -> Self {
        self * (delta.as_secs() as i64)
    }
    fn idt(self, delta: Duration) -> Self {
        self / (delta.as_secs() as i64)
    }
}
impl Dt for u64 {
    fn dt(self, delta: Duration) -> Self {
        self * delta.as_secs()
    }
    fn idt(self, delta: Duration) -> Self {
        self / delta.as_secs()
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Integrate<F, I> {
    integrand: F,
    acc: I,
    prev_time: Duration,
}
impl<F:Startable,I> Startable for Integrate<F, I>{
    fn start(&mut self, start: Duration){
        self.prev_time = start;
        self.integrand.start(start);
    }
}
impl<I, F: React<I>> React<I> for Integrate<F, F::Output>
where
    F::Output: Dt + Add<Output = F::Output> + Clone,
{
    type Output = F::Output;

    fn react(&mut self, input: I, time: Duration) -> F::Output {
        let delta = time - self.prev_time;
        self.prev_time = time;
        let next = self.acc.clone() + self.integrand.react(input, delta).dt(delta);
        self.acc = next;
        self.acc.clone()
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Derive<F, I> {
    derivand: F,
    prev: I,
    prev_time: Duration,
}
impl<F:Startable,I> Startable for Derive<F,I> {
    fn start(&mut self, start: Duration){
        self.prev_time = start;
        self.derivand.start(start);
    }   
}
impl<I, F: React<I>> React<I> for Derive<F, F::Output>
where
    F::Output: Dt + Sub<Output = F::Output> + Clone,
{
    type Output = F::Output;

    fn react(&mut self, input: I, time: Duration) -> F::Output {
        let delta = time - self.prev_time;
        self.prev_time = time;
        let next = self.derivand.react(input, delta);
        let dt = next.clone() - self.prev.clone();
        self.prev = next.clone();
        dt.idt(delta)
    }


}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub struct Done;

pub trait EventCarrier: Sized {
    type EventType;
    type ValueType;

    fn get_event(&self) -> &Event<Self::EventType>;

    fn get_value(&self) -> &Self::ValueType;

    fn to_event(self) -> Event<Self::EventType>;

    fn to_value(self) -> Self::ValueType;
}

impl<ValueType, EventType> EventCarrier for (ValueType, Event<EventType>) {
    type EventType = EventType;
    type ValueType = ValueType;

    fn get_event(&self) -> &Event<Self::EventType> {
        &self.1
    }

    fn get_value(&self) -> &Self::ValueType {
        &self.0
    }

    fn to_event(self) -> Event<Self::EventType> {
        self.1
    }

    fn to_value(self) -> Self::ValueType {
        self.0
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct ForS<R> {
    duration: Duration,
    start_time : Duration,
    wire: R,
}
impl<R:Startable> Startable for ForS<R>{
    fn start(&mut self, start: Duration){
        self.wire.start(start);
        self.start_time = start;
    }
}
impl<I, R: React<I>> React<I> for ForS<R> {
    type Output = (R::Output, Event<Done>);

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        let end_time =self.duration + self.start_time;
        let event = if self.duration + self.start_time <= time {
            let out = Event::Event(EventValue {
                value: Done,
                time: end_time
            });
            out
        } else {
            Event::NoEvent
        };

        let output = self.wire.react(input, time);

        (output, event)
    }
}
impl<R:Startable> Startable for ForForever<R>{
    fn start(&mut self, start: Duration){
        self.0.start(start);
    }
}
#[derive(Debug, Copy, Clone, react_traits)]
pub struct ForForever<R>(R);
impl<I, R: React<I>> React<I> for ForForever<R> {
    type Output = (R::Output, Event<Done>);

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        (self.0.react(input, time), Event::NoEvent)
    }


}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Then<R1, R2> {
    wire1: R1,
    wire2: R2,
    switched: bool,
}
impl<R1:Startable, R2:Startable> Startable for Then<R1, R2>{
    fn start(&mut self, start : Duration){
        self.wire1.start(start);
        self.wire2.start(start);
    }
}
impl<I, R1, R2> React<I> for Then<R1, R2>
where
    I: Clone,
    R1::Output: EventCarrier,
    R1: React<I>,
    R2: React<I, Output = R1::Output>,
{
    type Output = R2::Output;

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        if !self.switched {
            let output = self.wire1.react(input.clone(), time);
            let mut event_time = Duration::ZERO;
            match output.get_event() {
                Event::Event(EventValue { time, value: _ }) => {
                    self.switched = true;
                    event_time = *time;
                }
                Event::NoEvent => {}
            };
            if !self.switched {
                output
            } else {
                self.wire2.start(event_time);
                self.wire2.react(input, time)
            }
        } else {
            self.wire2.react(input, time)
        }
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct WLoop<R> {
    initial_state: R,
    wire: R,
}
impl<R:Startable> Startable for WLoop<R>{
    fn start(&mut self, start:Duration){
        self.wire.start(start);
    }
}
impl<I: Clone, R: React<I> + Clone> React<I> for WLoop<R>
where
    R::Output: EventCarrier,
{
    type Output = <R::Output as EventCarrier>::ValueType;

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        let mut output = self.wire.react(input.clone(), time);

        if let Event::Event(EventValue { time: event_time, value: _ }) = output.get_event() {
            self.wire = self.initial_state.clone();
            self.wire.start(*event_time);
            output = self.wire.react(input, time);
        };

        output.to_value()
    }
}

pub struct Became();
pub struct NoLonger();
pub struct Edge();

pub struct At<X>(Duration, X);
pub struct Never();
pub struct Now<X>(X);
pub struct Periodic<X>(Duration, X);
pub struct PeriodicList<X>(Duration, Vec<X>);

pub struct Once();
pub struct DropE();
pub struct DropWhile();
pub struct NotYet();
pub struct FilterE();
pub struct MergeE();

pub struct Cycle();
pub struct Until();
pub struct After();
pub struct Alternate();

pub struct Switch();
pub struct DSwitch();
pub struct KSwitch();

pub struct DKSwitch();
pub struct RSwitch();

macro_rules! reactive_bin_op {
    ($name:ident $trait:ident $op:tt) => {
        impl<R1: Startable,R2: Startable> Startable for $name<R1,R2>{
            fn start(&mut self, start: Duration){
                self.0.start(start);
                self.1.start(start);
            }
        }
        impl<I : Clone,R1: React<I>,R2: React<I>> React<I> for $name<R1,R2>
        where <R1 as React<I>>::Output : $trait<<R2 as React<I>>::Output>{
            type Output = <R1::Output as $trait<<R2 as React<I>>::Output>>::Output;

            fn react(&mut self, input: I, delta : Duration) -> Self::Output{
                self.0.react(input.clone(), delta) $op self.1.react(input, delta)
            }
        }
    }
}

macro_rules! reactive_un_op {
    ($name:ident $trait:ident $op:tt) => {
        impl<R: Startable> Startable for $name<R>{
            fn start(&mut self, start: Duration){
                self.0.start(start);
            }
        }
        impl<I, R: React<I>> React<I> for $name<R>
        where R::Output: $trait{
            type Output = <R::Output as $trait>::Output;

            fn react(&mut self, input: I, delta : Duration)-> Self::Output{
                $op self.0.react(input,delta)
            }
        }
    }
}
#[derive(Debug, Copy, Clone, react_traits)]
pub struct Sum<T1, T2>(T1, T2);
reactive_bin_op!(Sum Add +);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Difference<T1, T2>(T1, T2);
reactive_bin_op!(Difference Sub -);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Product<T1, T2>(T1, T2);
reactive_bin_op!(Product Mul *);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Quotient<T1, T2>(T1, T2);
reactive_bin_op!(Quotient Div /);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Negation<T>(T);
reactive_un_op!(Negation Neg -);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Ix<T, Index>(T, Index);
impl<R1: Startable, R2: Startable> Startable for Ix<R1,R2>{
    fn start(&mut self, start: Duration){
        self.0.start(start);
        self.1.start(start);
    }
}
impl<I: Clone, R1: React<I>, R2: React<I>> React<I> for Ix<R1, R2>
where
    <R1 as React<I>>::Output: Index<<R2 as React<I>>::Output>,
    <R1::Output as Index<<R2 as React<I>>::Output>>::Output: Clone,
{
    type Output = <R1::Output as Index<<R2 as React<I>>::Output>>::Output;

    fn react(&mut self, input: I, delta: Duration) -> Self::Output {
        self.0.react(input.clone(), delta)[self.1.react(input, delta)].clone()
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Dereference<T>(T);
impl<R:Startable> Startable for Dereference<R>{
    fn start(&mut self, start: Duration){
        self.0.start(start);
    }
}
impl<I, R: React<I>> React<I> for Dereference<R>
where
    R::Output: Deref,
    <R::Output as Deref>::Target: Clone,
{
    type Output = <R::Output as Deref>::Target;

    fn react(&mut self, input: I, delta: Duration) -> Self::Output {
        self.0.react(input, delta).clone()
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Remainder<T1, T2>(T1, T2);
reactive_bin_op!(Remainder Rem %);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct LeftShift<T1, T2>(T1, T2);
reactive_bin_op!(LeftShift Shl  << );

#[derive(Debug, Copy, Clone, react_traits)]
pub struct RightShift<T1, T2>(T1, T2);
reactive_bin_op!(RightShift Shr >> );

#[derive(Debug, Copy, Clone, react_traits)]
pub struct BitwiseAnd<T1, T2>(T1, T2);
reactive_bin_op!(BitwiseAnd BitAnd &);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct BitwiseOr<T1, T2>(T1, T2);
reactive_bin_op!(BitwiseOr BitOr |);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct BitwiseXor<T1, T2>(T1, T2);
reactive_bin_op!(BitwiseXor BitXor ^);

#[derive(Debug, Copy, Clone, react_traits)]
pub struct BitwiseNot<T>(T);
reactive_un_op!(BitwiseNot Not !);

pub enum StepType {
    Discrete,
    Continuous,
}
pub struct Clock {
    pub step_type: StepType,
    pub target_fps: f64,
}

impl Clock {
    pub fn run<W, F>(mut reactive: W, callback: F)
    where
        W: React<()>,
        F: Fn(W::Output) -> bool,
    {
        let mut prev = Instant::now();
        reactive.start(Duration::ZERO);

        loop {
            let now = Instant::now();
            let delta = now.duration_since(prev);
            prev = now;

            let output = reactive.react((), delta);

            if !callback(output) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    fn test_const() {
        assert_eq!(5, (5 as i32).react((), Duration::from_secs(1)))
    }

    #[test]
    fn test_integrate() {
        assert_eq!(
            10.0,
            Integrate {
                integrand: 5.0,
                acc: 0.0,
                prev_time: Duration::ZERO
            }
            .react((), Duration::from_secs(2))
        );
        let mut integral = Integrate {
            integrand: 5.0,
            acc: 0.0,
            prev_time: Duration::ZERO
        };
        integral.start(Duration::from_secs(1));
        assert_eq!(
            10.0,integral.react((), Duration::from_secs(3))
        );


    }

    #[test]
    fn test_split() {
        assert_eq!((5, 6), Split(5, 6).react((), Duration::new(0, 0)))
    }

    #[test]
    fn test_first() {
        assert_eq!((5, 6), First(5).react(((), 6), Duration::new(0, 0)))
    }

    #[test]
    fn test_second() {
        assert_eq!((5, 6), Second(6).react((5, ()), Duration::new(0, 0)))
    }

    #[test]
    fn test_func() {
        assert_eq!(5, Func(|x, _| x + 1).react(4, Duration::new(0, 0)))
    }

    #[test]
    fn test_fanout() {
        assert_eq!(
            (5, 6),
            FanOut(Func(|x, _| x + 1), Func(|x, _| x - 1)).react((4, 7), Duration::new(0, 0))
        )
    }

    #[test]
    fn test_pure() {
        assert_eq!("abc", Pure("abc").react((), Duration::new(0, 0)))
    }

    #[test]
    fn test_composed() {
        assert_eq!(
            13,
            Composed(
                Func(|x, _| x + 1),
                Integrate {
                    integrand: 6,
                    acc: 0,
                    prev_time : Duration::ZERO
                }
            )
            .react((), Duration::from_secs(2))
        )
    }

    #[test]
    fn test_add() {
        assert_eq!(3, (Func(|x, _| x + 1) + 1).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_sub() {
        assert_eq!(2, (Func(|x, _| x + 2) - 1).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_mul() {
        assert_eq!(6, (Func(|x, _| x + 2) * 2).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_div() {
        assert_eq!(3, (Func(|x, _| x + 5) / 2).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_rem() {
        assert_eq!(2, (Func(|x, _| x + 5) % 4).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_shl() {
        assert_eq!(2, (Func(|x, _| x + 3) >> 1).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_shr() {
        assert_eq!(8, (Func(|x, _| x + 3) << 1).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_bitand() {
        assert_eq!(
            false,
            (Func(|x, _| x > 3) & true).react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_bitor() {
        assert_eq!(
            true,
            (Func(|x, _| x < 3) | false).react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_xor() {
        assert_eq!(
            true,
            (Func(|x, _| x < 3) ^ false).react(1, Duration::new(0, 0))
        );
        assert_eq!(
            false,
            (Func(|x, _| x < 3) ^ true).react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_negation() {
        assert_eq!(-4, (-Func(|x, _| x + 3)).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_not() {
        assert_eq!(false, (!Func(|x, _| x < 3)).react(1, Duration::new(0, 0)));
    }

    #[test]
    fn test_fors() {
        let mut reactive = ForS {
            duration: Duration::from_secs(2),
            start_time : Duration::ZERO,
            wire: 5,
        };

        let (val1, event1) = reactive.react((), Duration::from_secs(1));

        assert_eq!(event1, Event::NoEvent);
        assert_eq!(val1, 5);

        let (val2, event2) = reactive.react((), Duration::from_secs(2));

        assert_eq!(
            event2,
            Event::Event(EventValue {
                time: Duration::from_secs(2),
                value: Done
            })
        );
        assert_eq!(val2, 5);
    }

    #[test]
    fn test_for_forever() {
        let mut reactive = ForForever(5);

        assert_eq!(
            reactive.react((), Duration::from_secs(10000)),
            (5, Event::NoEvent)
        );
        assert_eq!(
            reactive.react((), Duration::from_secs(20000)),
            (5, Event::NoEvent)
        );
        assert_eq!(
            reactive.react((), Duration::from_secs(30000)),
            (5, Event::NoEvent)
        );
    }

    #[test]
    fn test_then() {
        let wire1 = ForS {
            duration: Duration::from_secs_f32(1.0),
            start_time : Duration::ZERO,
            wire: 2,
        };
        let wire2 = ForS {
            duration: Duration::from_secs_f32(1.0),
            start_time : Duration::ZERO,
            wire: 3,
        };
        let mut reactive = Then {
            wire1,
            wire2,
            switched: false,
        };

        let val1 = reactive.react((), Duration::from_secs_f32(0.5));

        assert_eq!(val1, (2, Event::NoEvent));

        let val2 = reactive.react((), Duration::from_secs_f32(1.5));

        assert_eq!(val2, (3, Event::NoEvent));

        let val2 = reactive.react((), Duration::from_secs_f32(2.0));


        assert_eq!(val2, (3,Event::Event(EventValue{time: Duration::from_secs(2), value: Done})));
    }

    #[test]
    fn test_wloop() {
        let looped = Then {
            wire1: ForS {
                duration: Duration::from_secs_f32(1.0),
                start_time : Duration::ZERO,
                wire: 2,
            },
            wire2: ForS {
                duration: Duration::from_secs_f32(1.0),
                start_time : Duration::ZERO,
                wire: 3,
            },
            switched: false,
        };
        let mut wloop = WLoop {
            initial_state: looped.clone(),
            wire: looped,
        };

        let val1 = wloop.react((), Duration::from_secs_f32(0.5));
        assert_eq!(val1, 2);

        let val2 = wloop.react((), Duration::from_secs_f32(1.5));
        assert_eq!(val2, 3);

        let val3 = wloop.react((), Duration::from_secs_f32(2.5));
        assert_eq!(val3, 2);

        let val4 = wloop.react((), Duration::from_secs_f32(4.5)); // go around in 1 step
        assert_eq!(val4, 2);
    }
}
