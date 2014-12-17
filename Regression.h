#ifndef REGRESSION_INCLUDED__H
#define REGRESSION_INCLUDED__H

#include"LinearAlgebra.h"
#include"Model.h"

template< class X, class KernelFunction >
class KernelizedModel;

namespace Regression{

    LinearAlgebra::Matrix ordinaryLeastSquares( const LinearAlgebra::Matrix& X, const LinearAlgebra::Matrix& Y );

    LinearAlgebra::Matrix ridge( const LinearAlgebra::Matrix& X, const LinearAlgebra::Matrix& Y, double lambda );

    template< typename X, typename KernelFunction >
    inline LinearAlgebra::Matrix kernelMatrix( const X& x, const KernelFunction& kernel ){
        unsigned long n = x.size();
        LinearAlgebra::Matrix k( n, n );
        unsigned long i,j;
        for( j = 0 ; j < n ; ++j ){
            for( i = j ; i < n ; ++i ){
                double val = kernel( x[i], x[j] );
                k( i, j ) = val;
                k( j, i ) = val;
            }
        }
        return k;
    }

/*
    template< typename It, typename KernelFunction >
    inline LinearAlgebra::Matrix kernelMatrix( It start, It end, const KernelFunction& f_kernel ){

        unsigned long n = end - start;
        LinearAlgebra::Matrix k( n, n );
        unsigned long i,j;
        for( j = 0 ; j < n ; ++j ){
            for( i = j ; i < n ; ++i ){
                double val = f_kernel( *(start + i), *(start + j) );
                k( i, j ) = val;
                k( j, i ) = val;
            }
        }
        return k;

    }
*/

    template< class X, class KernelFunction >
    inline LinearAlgebra::Matrix kernelMatrix( const X& x0, const X& x1, const KernelFunction& f_kernel ){

        unsigned long n, m;
        n = x0.size();
        m = x1.size();
        LinearAlgebra::Matrix k( n, m );
        unsigned long i, j;
        for( j = 0 ; j < m ; ++j ){
            for( i = 0 ; i < n ; ++i ){
                k( i, j ) = f_kernel( x0[i], x1[j] );
            }
        }
        return k;
    }

    template< typename It, typename KernelFunction >
    inline LinearAlgebra::Matrix kernelMatrix( It start0, It end0, It start1, It end1, const KernelFunction& f_kernel ){

        unsigned long n, m;
        n = end0 - start0;
        m = end1 - start1;
        LinearAlgebra::Matrix k( n, m );
        unsigned long i,j;
        for( j = 0 ; j < m ; ++j ){
            for( i = 0 ; i < n ; ++i ){
                k( i, j ) = f_kernel( *(start0 + i), *(start1 + j) );
            }
        }
        return k;

    }

    template< typename X, typename KernelFunction>
    KernelizedModel<X,KernelFunction> ridge( const X& x, const LinearAlgebra::Vector& y, double lambda, const KernelFunction& kernel ){
        
        unsigned long n = x.size();

        //compute the kernel matrix
        LinearAlgebra::Matrix k = kernelMatrix( x, kernel );
        
        //add lambda to the diagonal
        for( unsigned long i = 0 ; i < n ; ++i ){
            k( i, i ) += lambda;
        }
        //center Ys
        LinearAlgebra::Vector y_centered = LinearAlgebra::center( y );
        //solve
        LinearAlgebra::Vector alphas = LinearAlgebra::solve( k, y_centered );
       //remove lambdas from kernel matrix
       for( unsigned long i = 0 ; i < n ; ++i ){
           k( i, i ) -= lambda;
       }
       double bias = y.mean() - k.colwise().mean() * alphas;

       return KernelizedModel<X,KernelFunction>( bias, alphas, x, kernel );

    }

    LinearModel ridge( LinearAlgebra::Matrix& k, const LinearAlgebra::Vector& y, double lambda ){
        
        unsigned long n = k.rows();
        for( unsigned long i = 0 ; i < n ; ++i ){
            k( i, i ) += lambda;
        }
        LinearAlgebra::Vector y_centered = LinearAlgebra::center( y );
        LinearAlgebra::Vector alphas = LinearAlgebra::solve( k, y_centered );
        for( unsigned long i = 0 ; i < n ; ++i ){
            k( i, i ) -= lambda;
        }
        double bias = y.mean() - k.colwise().mean() * alphas;

        return LinearModel( bias, alphas );

    }



}





#endif
