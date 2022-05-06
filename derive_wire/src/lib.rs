
extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2};
use quote::quote;
use syn::{parse_macro_input, DeriveInput,GenericParam};


#[proc_macro_derive(react_traits)]
pub fn react_traits(input: TokenStream) -> TokenStream{
    let input = parse_macro_input!(input as DeriveInput);

    let ident = input.ident;
    let generic_list : Vec<GenericParam> = input.generics.params.into_iter().collect();
    let maybe_where_clause = input.generics.where_clause;

    let simple_binary_trait = |trait_ :TokenStream2, output_: TokenStream2 , fn_ : TokenStream2| {

        quote!{
            impl<Other, #(#generic_list),* > #trait_<Other> for #ident<#(#generic_list),*>{
                type Output = #output_<Self,Other>;

                fn #fn_(self,other: Other) -> Self::Output{
                    #output_(self,other)
                }
            }
        }

    };

    let simple_binary_traits : Vec<TokenStream2> = vec![
        (quote!{Add},quote!{Sum},quote!{add}),
        (quote!{Mul},quote!{Product},quote!{mul}),
        (quote!{Div},quote!{Quotient},quote!{div}),
        (quote!{Sub},quote!{Difference},quote!{sub}),
        (quote!{Rem},quote!{Remainder},quote!{rem}),
        (quote!{Shl},quote!{LeftShift},quote!{shl}),
        (quote!{Shr},quote!{RightShift},quote!{shr}),
        (quote!{BitAnd},quote!{BitwiseAnd},quote!{bitand}),
        (quote!{BitOr},quote!{BitwiseOr},quote!{bitor}),
        (quote!{BitXor},quote!{BitwiseXor},quote!{bitxor}),
    ].into_iter().map(|(t1,t2,t3)| simple_binary_trait(t1,t2,t3)).collect();


    quote!{
        #(#simple_binary_traits) *

        impl<#(#generic_list),*> Neg for #ident<#(#generic_list),*> {
            type Output = Negation<Self>;

            fn neg(self) -> Self::Output{
                Negation(self)
            }

        }


        impl<#(#generic_list),*> Not for #ident<#(#generic_list),*> {
            type Output = BitwiseNot<Self>;

            fn not(self) -> Self::Output {
                BitwiseNot(self)
            }
        }

    }.into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
