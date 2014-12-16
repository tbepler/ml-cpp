#ifndef REGRESSION_INCLUDED__H
#define REGRESSION_INCLUDED__H

#include"LinearAlgebra.h"
#include<vector>

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

    template< typename X, typename Y, typename KernelFunction>
    LinearAlgebra::Matrix ridge( const X& x, const Y& y, double lambda, const KernelFunction& kernel ){
        
        unsigned long n = x.size();

        //compute the kernel matrix
        LinearAlgebra::Matrix k = kernelMatrix( x, kernel );
        
        //add lambda to the diagonal
        for( unsigned long i = 0 ; i < n ; ++i ){
            k( i, i ) += lambda;
        }

        //solve
        return LinearAlgebra::solve<LinearAlgebra::Matrix>( k, y );

    }


}





#endif
