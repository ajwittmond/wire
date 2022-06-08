extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, GenericParam};

#[proc_macro_derive(react_traits)]
pub fn react_traits(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let ident = input.ident;
    let generic_list: Vec<GenericParam> = input.generics.params.into_iter().collect();
    let maybe_where_clause = input.generics.where_clause;

    let simple_binary_trait = |trait_: TokenStream2, output_: TokenStream2, fn_: TokenStream2| {
        quote! {
            impl<Other, #(#generic_list),* > #trait_<Other> for #ident<#(#generic_list),*>
            where  #ident<#(#generic_list),*> : React,
            Other : IntoReactive<<#ident<#(#generic_list),*> as React>::Input>,
            <Self as React>::Output : #trait_<<
                    <
                        Other as IntoReactive<<#ident<#(#generic_list),*> as React>::Input>
                    >::Reactive as React>::Output
                >{
                type Output = #output_<
                        Self,
                        <
                            Other as IntoReactive<<#ident<#(#generic_list),*> as React>::Input>
                        >::Reactive
                    >;

                fn #fn_(self,other: Other) -> <Self as #trait_<Other>>::Output{
                    #output_(self,other.into_reactive())
                }
            }
        }
    };

    let simple_binary_traits: Vec<TokenStream2> = vec![
        (quote! {Add}, quote! {Sum}, quote! {add}),
        (quote! {Mul}, quote! {Product}, quote! {mul}),
        (quote! {Div}, quote! {Quotient}, quote! {div}),
        (quote! {Sub}, quote! {Difference}, quote! {sub}),
        (quote! {Rem}, quote! {Remainder}, quote! {rem}),
        (quote! {Shl}, quote! {LeftShift}, quote! {shl}),
        (quote! {Shr}, quote! {RightShift}, quote! {shr}),
        (quote! {BitAnd}, quote! {BitwiseAnd}, quote! {bitand}),
        (quote! {BitOr}, quote! {BitwiseOr}, quote! {bitor}),
        (quote! {BitXor}, quote! {BitwiseXor}, quote! {bitxor}),
    ]
    .into_iter()
    .map(|(t1, t2, t3)| simple_binary_trait(t1, t2, t3))
    .collect();

    quote! {
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


        impl<E,#(#generic_list),*> IntoReactive<E> for #ident<#(#generic_list),*>
        where #ident<#(#generic_list),*> : React<Input=E> {
            type Reactive = #ident<#(#generic_list),*>;

            fn into_reactive(self)-> Self::Reactive{
                self
            }
        }

    }
    .into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
