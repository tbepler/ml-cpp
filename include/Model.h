#ifndef MODEL_INCLUDED__H
#define MODEL_INCLUDED__H

#include"LinearAlgebra.h"
#include"Regression.h"
#include<iostream>
#include<string>
#include<vector>

namespace Regression{
    template< class X, class KernelFunction >
    inline LinearAlgebra::Matrix kernelMatrix( const X& x0, const X& x1, const KernelFunction& f_kernel ); 
}

template< class X >
class Model{
    public:
        double bias;
        LinearAlgebra::Vector weights;

        virtual LinearAlgebra::Vector predict( const X& x ) const = 0;
        virtual ~Model() { /* do nothing by default */ };
};

class LinearModel : public Model<LinearAlgebra::Matrix>{

    public:

        LinearModel( ) { };

        LinearModel( const double bias, const LinearAlgebra::Vector& weights){
            this->bias = bias;
            this->weights = weights;
        };

        ~LinearModel() { /* do nothing */ };

        LinearAlgebra::Vector predict( const LinearAlgebra::Matrix& x ) const{
            return x * this->weights + LinearAlgebra::constant( x.rows(), this->bias );
        };

};

template< class X, class KernelFunction >
class KernelizedModel : public Model<X>{

    public:
        X data;
        KernelFunction f_kernel;

        KernelizedModel() {};

        KernelizedModel( const double bias, const LinearAlgebra::Vector& alphas, const X& data, const KernelFunction& f_kernel ){
            this->bias = bias;
            this->weights = alphas;
            this->data = data;
            this->f_kernel = f_kernel;
        };

        ~KernelizedModel( ) { /*do nothing*/ };

        LinearAlgebra::Vector predict( const X& x ) const{
            
            LinearAlgebra::Matrix k = Regression::kernelMatrix( x, data, f_kernel );
            return k * this->weights + LinearAlgebra::constant( x.size(), this->bias ); 

        };


};



#endif
