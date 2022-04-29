#![allow(dead_code)]
extern crate derive_wire;

use derive_wire::react_traits;
use std::ops::{Mul,Add,Sub,Div,Index,Neg,Not,BitAnd,BitOr,BitXor,Deref,Shl,Shr,Rem};
use std::time::{Duration,Instant};

pub trait React<Input> {
    type Output;
    fn react(&mut self, input: Input, delta: Duration) -> Self::Output;
}

// impl<I,E,C : Copy> React<I,E> for C {
//     type Output = C;

//     fn react(&mut self, input: I, delta: E) -> Self::Output{
//         *self
//     }
// }
macro_rules! copy_react_impls{
    ($($type:ty),*) => {
        $( impl<I> React<I> for $type{
            type Output = Self;

            fn react(&mut self, _input : I, _delta: Duration) -> Self{
                self.clone()
            }
        } ) *
    }
}

copy_react_impls![String, &str, char, i8, i16, i32, i64, f32, f64, u8, u16, u32, u64, bool];

macro_rules! bin_op_trait_impl{
    ($name:ident $trait:ident $func:ident $otype:ident) => {
        impl<A> $trait<A> for $name
        where A : React<I>
        {
            type Output = $otype;

            fn $func(self, other : A) -> Self::Output{
                $otype(self,other)
            }

        }

    }
}

macro_rules! react_traits {

    ($name:ident) => {

        bin_op_trait_impl!($name Add add Sum);
        bin_op_trait_impl!($name Sub sub Difference);
        bin_op_trait_impl!($name Mul mul Product);
        bin_op_trait_impl!($name Div div Quotient);



    }
}

#[derive(Debug, Copy, Clone)]
pub enum Event<E> {
    Event(E),
    NoEvent,
}
impl<E> Event<E>{

    fn map<E2,F : FnOnce(E) -> E2>(self,f : F) -> Event<E2>{
        use crate::Event::*;
        match self{
            Event(e) => Event(f(e)),
            NoEvent => NoEvent
        }
    }


    fn merge(self, other : Event<E>) -> Event<E>{
        use crate::Event::*;
        match self{
            NoEvent => other,
            Event(e) => Event(e)
        }
    }

}

#[derive(Debug, Copy, Clone, react_traits)]
struct Func<F>(F);
impl<I, O, F: Fn(I, Duration) -> O> React<I> for Func<F> {
    type Output = O;

    fn react(&mut self, input: I, delta: Duration) -> Self::Output {
        self.0(input, delta)
    }
}



#[derive(Debug, Copy, Clone)]
struct StatefulFunc<F,S>(F,S);
impl<I, O, S, F: Fn((I,&mut S), Duration) -> O> React<I> for StatefulFunc<F,S> {
    type Output = O;

    fn react(&mut self, input: I, delta: Duration) -> Self::Output {
        self.0((input,&mut self.1), delta)
    }
}

#[derive(Debug, Copy, Clone)]
struct First<F>(F);

impl<I, T, F: React<I>> React<(I, T)> for First<F> {
    type Output = (F::Output, T);

    fn react(&mut self, input: (I, T), delta: Duration) -> Self::Output {
        (self.0.react(input.0, delta), input.1)
    }
}

#[derive(Debug, Copy, Clone)]
struct Second<F>(F);

impl<I, T, F: React<I>> React<(T, I)> for Second<F> {
    type Output = (T, F::Output);

    fn react(&mut self, input: (T, I), delta: Duration) -> Self::Output {
        (input.0, self.0.react(input.1, delta))
    }
}

#[derive(Debug, Copy, Clone)]
struct Composed<F1, F2>(F1, F2);

impl<I,  F1: React<F2::Output>, F2: React<I>> React<I> for Composed<F1, F2> {
    type Output = F1::Output;

    fn react(&mut self, input: I, delta: Duration) -> Self::Output {
        self.0.react(self.1.react(input, delta), delta)
    }
}

#[derive(Debug, Copy, Clone)]
struct FanOut<F1, F2>(F1, F2);
impl<I1, I2,  F1: React<I1>, F2: React<I2>> React<(I1, I2)> for FanOut<F1, F2> {
    type Output = (F1::Output, F2::Output);

    fn react(&mut self, (i1, i2): (I1, I2), delta: Duration) -> Self::Output {
        (self.0.react(i1, delta), self.1.react(i2, delta))
    }
}

#[derive(Debug, Copy, Clone)]
struct Split<F1, F2>(F1, F2);
impl<I:Clone, F1: React<I>, F2: React<I>> React<I> for Split<F1, F2> {
    type Output = (F1::Output, F2::Output);

    fn react(&mut self, input: I, delta: Duration) -> Self::Output {
        (self.0.react(input.clone(), delta), self.1.react(input, delta))
    }
}

struct Pure<C>(C);
impl<I,C: Clone> React<I> for Pure<C>{
    type Output = C;

    fn react(&mut self, _input: I, _delta : Duration) -> C{
        self.0.clone()
    }
}

trait Dt{
    fn dt(self,delta : Duration) -> Self;
    fn idt(self,delta : Duration) -> Self;
}

impl Dt for f32{
    fn dt(self,delta : Duration) -> Self{
        self*delta.as_secs_f32()
    }
    fn idt(self,delta : Duration) -> Self{
        self/delta.as_secs_f32()
    }
}
impl Dt for f64{
    fn dt(self,delta : Duration) -> Self{
        self*delta.as_secs_f64()
    }
    fn idt(self,delta : Duration) -> Self{
        self/delta.as_secs_f64()
    }
}
impl Dt for i32{
    fn dt(self,delta : Duration) -> Self{
        self*(delta.as_secs() as i32)
    }
    fn idt(self,delta : Duration) -> Self{
        self/(delta.as_secs() as i32)
    }
}
impl Dt for u32{
    fn dt(self,delta : Duration) -> Self{
        self*(delta.as_secs() as u32)
    }
    fn idt(self,delta : Duration) -> Self{
        self/(delta.as_secs() as u32)
    }
}
impl Dt for i64{
    fn dt(self,delta : Duration) -> Self{
        self*(delta.as_secs() as i64)
    }
    fn idt(self,delta : Duration) -> Self{
        self/(delta.as_secs() as i64)
    }
}
impl Dt for u64{
    fn dt(self,delta : Duration) -> Self{
        self*delta.as_secs()
    }
    fn idt(self,delta : Duration) -> Self{
        self/delta.as_secs()
    }
}

#[derive(Debug, Copy, Clone)]
struct Integrate<F,I>{integrand: F, acc:I}
impl<I,F: React<I>> React<I> for Integrate<F,F::Output>
where F::Output : Dt + Add<Output = F::Output> + Clone
{
    type Output = F::Output;

    fn react(&mut self, input: I, delta: Duration) -> F::Output{
        let next  = self.acc.clone() +  self.integrand.react(input,delta).dt(delta);
        self.acc = next;
        self.acc.clone()
    }
}

#[derive(Debug, Copy, Clone)]
struct Derive<F,I>{integrand: F, prev:I}
impl<I,F: React<I>> React<I> for Derive<F,F::Output>
where F::Output : Dt + Sub<Output = F::Output> + Clone
{
    type Output = F::Output;

    fn react(&mut self, input: I, delta: Duration) -> F::Output{
        let next  = self.integrand.react(input,delta);
        let dt = next.clone() - self.prev.clone();
        self.prev =  next.clone();
        dt.idt(delta)
    }
}

struct Became();
struct NoLonger();
struct Edge();

struct At<X>(Duration,X);
struct Never();
struct Now<X>(X);
struct Periodic<X>(Duration,X);
struct PeriodicList<X>(Duration,Vec<X>);

struct Once();
struct DropE();
struct DropWhile();
struct NotYet();
struct FilterE();
struct MergeE();

struct Cycle();
struct Until();
struct After();
struct Alternate();

struct Switch();
struct DSwitch();
struct KSwitch();

struct DKSwitch();
struct RSwitch();


macro_rules! reactive_bin_op {
    ($name:ident $trait:ident $op:tt) => {
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
        impl<I, R: React<I>> React<I> for $name<R>
        where R::Output: $trait{
            type Output = <R::Output as $trait>::Output;

            fn react(&mut self, input: I, delta : Duration)-> Self::Output{
                $op self.0.react(input,delta)
            }
        }
    }
}
#[derive(Debug, Copy, Clone)]
struct Sum<T1,T2>(T1,T2);
reactive_bin_op!(Sum Add +);

#[derive(Debug, Copy, Clone)]
struct Difference<T1,T2>(T1,T2);
reactive_bin_op!(Difference Sub -);

#[derive(Debug, Copy, Clone)]
struct Product<T1,T2>(T1,T2);
reactive_bin_op!(Product Mul *);

#[derive(Debug, Copy, Clone)]
struct Quotient<T1,T2>(T1,T2);
reactive_bin_op!(Quotient Div /);

#[derive(Debug, Copy, Clone)]
struct Negations<T>(T);
reactive_un_op!(Negations Neg -);

#[derive(Debug, Copy, Clone)]
struct Ix<T1,T2>(T1,T2);
impl<I : Clone,R1: React<I>,R2: React<I>> React<I> for Ix<R1,R2>
where <R1 as React<I>>::Output : Index<<R2 as React<I>>::Output>,
      <R1::Output as Index<<R2 as React<I>>::Output>>::Output : Clone
{
    type Output = <R1::Output as Index<<R2 as React<I>>::Output>>::Output;

    fn react(&mut self, input: I, delta : Duration) -> Self::Output{
        self.0.react(input.clone(), delta)[self.1.react(input, delta)].clone()
    }
}

#[derive(Debug, Copy, Clone)]
struct Dereference<T>(T);
impl<I, R: React<I>> React<I> for Dereference<R>
where R::Output: Deref,
      <R::Output as Deref>::Target: Clone {
    type Output = <R::Output as Deref>::Target;

    fn react(&mut self, input: I, delta : Duration)-> Self::Output{
        self.0.react(input,delta).clone()
    }
}

#[derive(Debug, Copy, Clone)]
struct Remainder<T1,T2>(T1,T2);
reactive_bin_op!(Remainder Rem %);

#[derive(Debug, Copy, Clone)]
struct LeftShift<T1,T2>(T1,T2);
reactive_bin_op!(LeftShift Shl  << );

#[derive(Debug, Copy, Clone)]
struct RightShift<T1,T2>(T1,T2);
reactive_bin_op!(RightShift Shr >> );

#[derive(Debug, Copy, Clone)]
struct BitwiseAnd<T1,T2>(T1,T2);
reactive_bin_op!(BitwiseAnd BitAnd &);

#[derive(Debug, Copy, Clone)]
struct BitwiseOr<T1,T2>(T1,T2);
reactive_bin_op!(BitwiseOr BitOr |);

#[derive(Debug, Copy, Clone)]
struct BitwiseXor<T1,T2>(T1,T2);
reactive_bin_op!(BitwiseXor BitXor ^);


#[derive(Debug, Copy, Clone)]
struct BitwiseNot<T>(T);
reactive_un_op!(BitwiseNot Not !);


pub enum StepType{
    Discrete,
    Continuous
}
pub struct Clock{
    pub step_type : StepType,
    pub target_fps : f64,
}


impl Clock{

    pub fn run<W,F>(mut reactive : W, callback: F)
    where W : React<()>,
          F : Fn(W::Output) -> bool
    {
        let mut prev = Instant::now();

        loop{
            let now  = Instant::now();
            let delta = now.duration_since(prev);
            prev = now;

            let output = reactive.react((),delta);

            if ! callback(output){
                break
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
        assert_eq!(10.0, Integrate{integrand: 5.0, acc: 0.0}.react((),Duration::from_secs(2)))
    }

    #[test]
    fn test_split() {
        assert_eq!((5,6), Split(5,6).react((),Duration::new(0,0)))
    }

    #[test]
    fn test_first(){
        assert_eq!((5,6), First(5).react(((),6),Duration::new(0,0)))
    }

    #[test]
    fn test_second(){
        assert_eq!((5,6), Second(6).react((5,()),Duration::new(0,0)))
    }

    #[test]
    fn test_func(){
        assert_eq!(5,Func(|x ,_ | x + 1).react(4,Duration::new(0,0)) )

    }

    #[test]
    fn test_fanout(){
        assert_eq!((5,6), FanOut(Func(|x,_| x +1), Func(|x,_| x - 1)).react((4,7),Duration::new(0,0)))
    }

    #[test]
    fn test_pure(){
        assert_eq!("abc", Pure("abc").react((),Duration::new(0,0)))
    }

    #[test]
    fn test_composed(){
        assert_eq!(13, Composed(Func(|x,_| x+ 1), Integrate{integrand: 6, acc: 0}).react((),Duration::from_secs(2)))
    }

    #[test]
    fn test_add(){
        assert_eq!(3,(Func(|x,_| x+1) + 1).react(1,Duration::new(0,0)));

    }
}
