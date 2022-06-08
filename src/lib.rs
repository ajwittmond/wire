#![allow(dead_code)]
extern crate derive_wire;

use derive_wire::react_traits;
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitOr, BitXor, Deref, Div, Index, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use std::time::{Duration, Instant};



pub trait Rebind<E> {
    type Type;
}

// Trait to simulate functor like behaviors, while being
// slightly more permissive
// this is taken from the "functional" rust package which
// seemed to unstable to be added as a dependency
pub trait FMap<E1> {
    fn fmap<E2, F: FnOnce(E1) -> E2>(self, func: F) -> Self::Type
    where
        Self: Rebind<E2>;
}

pub trait Startable {
    /// establish a time offset for wires that act relative to a start time
    fn start(&mut self, _time: Duration) {}
}

pub trait React: Sized + Startable {
    type Input;
    type Output;
    fn react(&mut self, input: Self::Input, time: Duration) -> Self::Output;
}

pub trait IntoReactive<Input>{
    type Reactive : React<Input=Input>;

    fn into_reactive(self) -> Self::Reactive;
}

pub trait ReactFunctions: Startable + Sized {
    fn after(self) {
        todo!()
    }

    fn alternate(self) {
        todo!()
    }

    fn at(self, duration: Duration) {
        todo!()
    }

    fn fors<T: Into<f64>>(self, seconds: T) -> ForS<Self> {
        ForS {
            duration: Duration::from_secs_f64(seconds.into()),
            start_time: Duration::ZERO,
            wire: self,
        }
    }

    fn to<R>(self, other: R) -> Composed<Self, R> {
        Composed(self, other)
    }

    fn then<R>(self, other: R) -> Then<Self, R> {
        Then {
            wire1: self,
            wire2: other,
            switched: false,
        }
    }

    fn wloop(self) -> WLoop<Self>
    where
        Self: Clone,
    {
        WLoop {
            initial_state: self.clone(),
            wire: self,
        }
    }
}

pub fn at() {}

pub fn wloop<R: ReactFunctions + Clone>(reactive: R) -> WLoop<R> {
    reactive.wloop()
}

pub fn fors<R: ReactFunctions, T: Into<f64>>(duration: T, reactive: R) -> ForS<R> {
    reactive.fors(duration)
}

impl<A: Startable + Sized> ReactFunctions for A {}



#[derive(Debug, Copy, Clone, react_traits)]
pub struct Const<Input,Data>{
    phantom_input : PhantomData<Input>,
    pub data : Data
}

impl<Input,Data> Startable for Const<Input,Data>{}
impl<Input,Data : Clone> React for Const<Input,Data>{
    type Input = Input;
    type Output = Data;

    fn react(&mut self, _in : Input, _time: Duration) -> Data{
        self.data.clone()
    }
}


// impl<I,E,C : Copy> React<I,E> for C {
//     type Output = C;

//     fn react(&mut self, input: I, time: E) -> Self::Output{
//         *self
//     }
// }
macro_rules! copy_into_react_impls{
    ($($type:ty),*) => {
        $(
            impl<Input> IntoReactive<Input> for $type{
                type Reactive = Const<Input,$type>;

                fn into_reactive(self) -> Self::Reactive{
                    Const{phantom_input : PhantomData, data: self}
                }
            }
        ) *
    }
}

copy_into_react_impls![String, & 'static str, char, i8, i16, i32, i64, f32, f64, u8, u16, u32, u64, bool];

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct EventValue<E> {
    pub value: E,
    pub time: Duration,
}

impl<E1, E2> Rebind<E2> for EventValue<E1> {
    type Type = EventValue<E2>;
}
impl<E1> FMap<E1> for EventValue<E1> {
    fn fmap<E2, F: FnOnce(E1) -> E2>(self, f: F) -> <EventValue<E1> as Rebind<E2>>::Type {
        let x = f(self.value);
        EventValue {
            value: x,
            time: self.time,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum Event<E> {
    Event(EventValue<E>),
    NoEvent,
}

impl<E> Event<E> {
    fn merge(self, other: Event<E>) -> Event<E> {
        use crate::Event::*;
        match self {
            NoEvent => other,
            Event(e) => Event(e),
        }
    }
}
impl<E1, E2> Rebind<E2> for Event<E1> {
    type Type = Event<E2>;
}
impl<E1> FMap<E1> for Event<E1> {
    fn fmap<E2, F: FnOnce(E1) -> E2>(self, f: F) -> <Event<E1> as Rebind<E2>>::Type {
        use crate::Event::*;
        if let Event(e) = self {
            Event(e.fmap(f))
        } else {
            NoEvent
        }
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Func<I , F > {
    pub phantom: PhantomData<I>,
    pub func: F,
}
impl<I, F> Startable for Func<I, F> {}
impl<I, O, F: Fn(I, Duration) -> O> React for Func<I, F> {
    type Input = I;
    type Output = O;

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        (self.func)(input, time)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct StatefulFunc<I, F, S> {
    pub phantom: PhantomData<I>,
    pub func: F,
    pub state: S,
}
impl<I, F, S> Startable for StatefulFunc<I, F, S> {}
impl<I, O, S, F: Fn((I, &mut S), Duration) -> O> React for StatefulFunc<I, F, S> {
    type Input = I;
    type Output = O;

    fn react(&mut self, input: I, time: Duration) -> Self::Output {
        (self.func)((input, &mut self.state), time)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct First<F, T>(F, PhantomData<T>);

impl<T, F: Startable> Startable for First<F, T> {
    fn start(&mut self, start: Duration) {
        self.0.start(start)
    }
}
impl<T, F: React> React for First<F, T> {
    type Input = (F::Input, T);
    type Output = (F::Output, T);

    fn react(&mut self, input: (F::Input, T), time: Duration) -> Self::Output {
        (self.0.react(input.0, time), input.1)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Second<T, F>(PhantomData<T>, F);

impl<T, F: Startable> Startable for Second<T, F> {
    fn start(&mut self, start: Duration) {
        self.1.start(start)
    }
}
impl<T, F: React> React for Second<T, F> {
    type Input = (T, F::Input);
    type Output = (T, F::Output);

    fn react(&mut self, input: (T, F::Input), time: Duration) -> Self::Output {
        (input.0, self.1.react(input.1, time))
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Composed<F1, F2>(F1, F2);

impl<F1: Startable, F2: Startable> Startable for Composed<F1, F2> {
    fn start(&mut self, start: Duration) {
        self.0.start(start);
        self.1.start(start);
    }
}
impl<F1: React<Input = F2::Output>, F2: React> React for Composed<F1, F2> {
    type Input = F2::Input;
    type Output = F1::Output;

    fn react(&mut self, input: Self::Input, time: Duration) -> Self::Output {
        self.0.react(self.1.react(input, time), time)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct FanOut<F1, F2>(F1, F2);
impl<F1: Startable, F2: Startable> Startable for FanOut<F1, F2> {
    fn start(&mut self, start: Duration) {
        self.0.start(start);
        self.1.start(start);
    }
}
impl<F1: React, F2: React> React for FanOut<F1, F2> {
    type Input = (F1::Input, F2::Input);
    type Output = (F1::Output, F2::Output);

    fn react(&mut self, (i1, i2): Self::Input, time: Duration) -> Self::Output {
        (self.0.react(i1, time), self.1.react(i2, time))
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Split<F1, F2>(F1, F2);
impl<F1: Startable, F2: Startable> Startable for Split<F1, F2> {
    fn start(&mut self, start: Duration) {
        self.0.start(start);
        self.1.start(start);
    }
}
impl<F1: React, F2: React<Input = F1::Input>> React for Split<F1, F2>
where
    F1::Input: Clone,
{
    type Input = F1::Input;
    type Output = (F1::Output, F2::Output);

    fn react(&mut self, input: Self::Input, time: Duration) -> Self::Output {
        (self.0.react(input.clone(), time), self.1.react(input, time))
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Pure<C>(pub C);
impl<C> Startable for Pure<C> {}
impl<C: Clone> React for Pure<C> {
    type Input = ();
    type Output = C;

    fn react(&mut self, _input: (), _time: Duration) -> C {
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
impl<F: Startable, I> Startable for Integrate<F, I> {
    fn start(&mut self, start: Duration) {
        self.prev_time = start;
        self.integrand.start(start);
    }
}
impl<F: React> React for Integrate<F, F::Output>
where
    F::Output: Dt + Add<Output = F::Output> + Clone,
{
    type Input = F::Input;
    type Output = F::Output;

    fn react(&mut self, input: Self::Input, time: Duration) -> F::Output {
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
impl<F: Startable, I> Startable for Derive<F, I> {
    fn start(&mut self, start: Duration) {
        self.prev_time = start;
        self.derivand.start(start);
    }
}
impl<F: React> React for Derive<F, F::Output>
where
    F::Output: Dt + Sub<Output = F::Output> + Clone,
{
    type Output = F::Output;
    type Input = F::Input;

    fn react(&mut self, input: F::Input, time: Duration) -> F::Output {
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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub struct EventAndValue<ValueType, EventType>(pub ValueType, pub Event<EventType>);

impl<ValueType, EventType> EventCarrier for EventAndValue<ValueType, EventType> {
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
    start_time: Duration,
    wire: R,
}
impl<R: Startable> Startable for ForS<R> {
    fn start(&mut self, start: Duration) {
        self.wire.start(start);
        self.start_time = start;
    }
}
impl<R: React> React for ForS<R> {
    type Input = R::Input;
    type Output = EventAndValue<R::Output, Done>;

    fn react(&mut self, input: Self::Input, time: Duration) -> Self::Output {
        let end_time = self.duration + self.start_time;
        let event = if self.duration + self.start_time <= time {
            let out = Event::Event(EventValue {
                value: Done,
                time: end_time,
            });
            out
        } else {
            Event::NoEvent
        };

        let output = self.wire.react(input, time);

        EventAndValue(output, event)
    }
}
impl<R: Startable> Startable for ForForever<R> {
    fn start(&mut self, start: Duration) {
        self.0.start(start);
    }
}
#[derive(Debug, Copy, Clone, react_traits)]
pub struct ForForever<R>(R);
impl<R: React> React for ForForever<R> {
    type Input = R::Input;
    type Output = (R::Output, Event<Done>);

    fn react(&mut self, input: Self::Input, time: Duration) -> Self::Output {
        (self.0.react(input, time), Event::NoEvent)
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Then<R1, R2> {
    wire1: R1,
    wire2: R2,
    switched: bool,
}
impl<R1: Startable, R2: Startable> Startable for Then<R1, R2> {
    fn start(&mut self, start: Duration) {
        self.wire1.start(start);
        self.wire2.start(start);
    }
}
impl<R1, R2> React for Then<R1, R2>
where
    R1::Output: EventCarrier,
    R1: React,
    R2: React<Input = R1::Input, Output = R1::Output>,
    R1::Input: Clone,
{
    type Input = R2::Input;
    type Output = R2::Output;

    fn react(&mut self, input: Self::Input, time: Duration) -> Self::Output {
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
impl<R: Startable> Startable for WLoop<R> {
    fn start(&mut self, start: Duration) {
        self.wire.start(start);
    }
}
impl<R: React + Clone> React for WLoop<R>
where
    R::Output: EventCarrier,
    R::Input: Clone,
{
    type Input = R::Input;
    type Output = <R::Output as EventCarrier>::ValueType;

    fn react(&mut self, input: Self::Input, time: Duration) -> Self::Output {
        let mut output = self.wire.react(input.clone(), time);

        if let Event::Event(EventValue {
            time: event_time,
            value: _,
        }) = output.get_event()
        {
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
        impl< R1: React,R2: React<Input=R1::Input>> React for $name<R1,R2>
            where R1::Input : Clone,
                  R1::Output : $trait<R2::Output>
        {
            type Input = R1::Input;
            type Output = <R1::Output as $trait<<R2 as React>::Output>>::Output;

            fn react(&mut self, input: Self::Input, delta : Duration) -> Self::Output{
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
        impl<R: React> React for $name<R>
        where R::Output: $trait{
            type Input = R::Input;
            type Output = <R::Output as $trait>::Output;

            fn react(&mut self, input: Self::Input, delta : Duration)-> Self::Output{
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
impl<R1: Startable, R2: Startable> Startable for Ix<R1, R2> {
    fn start(&mut self, start: Duration) {
        self.0.start(start);
        self.1.start(start);
    }
}
impl<R1: React, R2: React<Input = R1::Input>> React for Ix<R1, R2>
where
    R1::Input: Clone,
    <R1 as React>::Output: Index<<R2 as React>::Output>,
    <R1::Output as Index<<R2 as React>::Output>>::Output: Clone,
{
    type Input = R1::Input;
    type Output = <R1::Output as Index<<R2 as React>::Output>>::Output;

    fn react(&mut self, input: Self::Input, delta: Duration) -> Self::Output {
        self.0.react(input.clone(), delta)[self.1.react(input, delta)].clone()
    }
}

#[derive(Debug, Copy, Clone, react_traits)]
pub struct Dereference<T>(T);
impl<R: Startable> Startable for Dereference<R> {
    fn start(&mut self, start: Duration) {
        self.0.start(start);
    }
}
impl<R: React> React for Dereference<R>
where
    R::Output: Deref,
    <R::Output as Deref>::Target: Clone,
{
    type Input = R::Input;
    type Output = <R::Output as Deref>::Target;

    fn react(&mut self, input: Self::Input, delta: Duration) -> Self::Output {
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
    pub fn run<W, F>(&self, mut reactive: W, callback: F)
    where
        W: React<Input = ()>,
        F: Fn(W::Output) -> bool,
    {
        let mut prev = Instant::now();
        let mut steps_waited = 0;
        reactive.start(Duration::ZERO);

        loop {
            let now = Instant::now();
            let delta = now.duration_since(prev);
            let target_timestep = Duration::from_secs_f64(1.0 / self.target_fps);
            let average_delta = delta / (steps_waited + 1);
            if delta + average_delta >= target_timestep {
                prev = now;

                let step = match self.step_type {
                    StepType::Continuous => delta,
                    StepType::Discrete => target_timestep,
                };

                let output = reactive.react((), step);

                if !callback(output) {
                    break;
                }
            } else {
                steps_waited += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    fn test_const() {
        assert_eq!(5, (5 as i32).into_reactive().react((), Duration::from_secs(1)))
    }

    #[test]
    fn test_integrate() {
        assert_eq!(
            10.0,
            Integrate {
                integrand: 5.0.into_reactive(),
                acc: 0.0,
                prev_time: Duration::ZERO
            }
            .react((), Duration::from_secs(2))
        );
        let mut integral = Integrate {
            integrand: 5.0.into_reactive(),
            acc: 0.0,
            prev_time: Duration::ZERO,
        };
        integral.start(Duration::from_secs(1));
        assert_eq!(10.0, integral.react((), Duration::from_secs(3)));
    }

    #[test]
    fn test_split() {
        assert_eq!((5, 6), Split(5.into_reactive(), 6.into_reactive()).react((), Duration::new(0, 0)))
    }

    #[test]
    fn test_first() {
        assert_eq!(
            (5, 6),
            First(5.into_reactive(), PhantomData).react(((), 6), Duration::new(0, 0))
        )
    }

    #[test]
    fn test_second() {
        assert_eq!(
            (5, 6),
            Second(PhantomData, 6.into_reactive()).react((5, ()), Duration::new(0, 0))
        )
    }

    #[test]
    fn test_func() {
        assert_eq!(
            5,
            Func {
                phantom: PhantomData,
                func: |x, _| x + 1
            }
            .react(4, Duration::new(0, 0))
        )
    }

    #[test]
    fn test_fanout() {
        assert_eq!(
            (5, 6),
            FanOut(
                Func {
                    phantom: PhantomData,
                    func: |x, _| x + 1
                },
                Func {
                    phantom: PhantomData,
                    func: |x, _| x - 1
                }
            )
            .react((4, 7), Duration::new(0, 0))
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
                Func {
                    phantom: PhantomData,
                    func: |x, _| x + 1
                },
                Integrate {
                    integrand: 6.into_reactive(),
                    acc: 0,
                    prev_time: Duration::ZERO
                }
            )
            .react((), Duration::from_secs(2))
        )
    }

    #[test]
    fn test_add() {
        assert_eq!(
            3,
            (Func {
                phantom: PhantomData::<i32>,
                func: |x, _| x + 1
            } + 1 )
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_sub() {
        assert_eq!(
            2,
            (Func {
                phantom: PhantomData,
                func: |x, _| x + 2
            } - 1)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_mul() {
        assert_eq!(
            6,
            (Func {
                phantom: PhantomData,
                func: |x, _| x + 2
            } * 2)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_div() {
        assert_eq!(
            3,
            (Func {
                phantom: PhantomData,
                func: |x, _| x + 5
            } / 2)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_rem() {
        assert_eq!(
            2,
            (Func {
                phantom: PhantomData,
                func: |x, _| x + 5
            } % 4)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_shl() {
        assert_eq!(
            2,
            (Func {
                phantom: PhantomData,
                func: |x, _| x + 3
            } >> 1)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_shr() {
        assert_eq!(
            8,
            (Func {
                phantom: PhantomData,
                func: |x, _| x + 3
            } << 1)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_bitand() {
        assert_eq!(
            false,
            (Func {
                phantom: PhantomData,
                func: |x, _| x > 3
            } & true)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_bitor() {
        assert_eq!(
            true,
            (Func {
                phantom: PhantomData,
                func: |x, _| x < 3
            } | false)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_xor() {
        assert_eq!(
            true,
            (Func {
                phantom: PhantomData,
                func: |x, _| x < 3
            } ^ false)
                .react(1, Duration::new(0, 0))
        );
        assert_eq!(
            false,
            (Func {
                phantom: PhantomData,
                func: |x, _| x < 3
            } ^ true)
                .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_negation() {
        assert_eq!(
            -4,
            (-Func {
                phantom: PhantomData,
                func: |x, _| x + 3
            })
            .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_not() {
        assert_eq!(
            false,
            (!Func {
                phantom: PhantomData,
                func: |x, _| x < 3
            })
            .react(1, Duration::new(0, 0))
        );
    }

    #[test]
    fn test_fors() {
        let mut reactive = ForS {
            duration: Duration::from_secs(2),
            start_time: Duration::ZERO,
            wire: 5.into_reactive(),
        };

        let EventAndValue(val1, event1) = reactive.react((), Duration::from_secs(1));

        assert_eq!(event1, Event::NoEvent);
        assert_eq!(val1, 5);

        let EventAndValue(val2, event2) = reactive.react((), Duration::from_secs(2));

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
        let mut reactive = ForForever(5.into_reactive());

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
            start_time: Duration::ZERO,
            wire: 2.into_reactive(),
        };
        let wire2 = ForS {
            duration: Duration::from_secs_f32(1.0),
            start_time: Duration::ZERO,
            wire: 3.into_reactive(),
        };
        let mut reactive = Then {
            wire1,
            wire2,
            switched: false,
        };

        let val1 = reactive.react((), Duration::from_secs_f32(0.5));

        assert_eq!(val1, EventAndValue(2, Event::NoEvent));

        let val2 = reactive.react((), Duration::from_secs_f32(1.5));

        assert_eq!(val2, EventAndValue(3, Event::NoEvent));

        let val2 = reactive.react((), Duration::from_secs_f32(2.0));

        assert_eq!(
            val2,
            EventAndValue(
                3,
                Event::Event(EventValue {
                    time: Duration::from_secs(2),
                    value: Done
                })
            )
        );
    }

    #[test]
    fn test_wloop() {
        let looped = Then {
            wire1: ForS {
                duration: Duration::from_secs_f32(1.0),
                start_time: Duration::ZERO,
                wire: 2.into_reactive(),
            },
            wire2: ForS {
                duration: Duration::from_secs_f32(1.0),
                start_time: Duration::ZERO,
                wire: 3.into_reactive(),
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
