#ifndef CROSSVAL_INCLUDED__H
#define CROSSVAL_INCLUDED__H

#include"LinearAlgebra.h"
#include"Regression.h"
#include"Model.h"
#include<vector>
#include<algorithm>
#include<cmath>

struct CrossValidation{

    std::vector< double > errors;
    std::vector< double > parameters;
    double selected;

};
    
template< class T, class KernelFunction, class TrainFunction, class ErrorFunction >
CrossValidation crossValidate( std::vector<T>& x, LinearAlgebra::Vector& y, std::vector<double> parameters, const KernelFunction& f_kernel, const TrainFunction& f_train, const ErrorFunction& f_error, unsigned long kfold ){

    unsigned long n = y.size();

    //randomly permute x and y
    std::vector< unsigned long > indices;
    for( unsigned long i = 0 ; i < n ; ++i ){
        indices.push_back( i );
    }
    std::random_shuffle( indices.begin(), indices.end() );
    std::vector<T> x_permuted( n );
    LinearAlgebra::Vector y_permuted( n );
    for( unsigned long i = 0 ; i < n ; ++i ){
        x_permuted[i] = x[ indices[i] ];
        y_permuted[i] = y[ indices[i] ];
    }

    std::vector< double > errors( parameters.size(), 0 );
    //perform cross validation
    for( unsigned long j = 0 ; j < kfold ; ++j ){
        unsigned long start = j * n / kfold;
        unsigned long end = ( j + 1 ) * n / kfold;
        std::vector<T> x_train;
        std::vector<T> x_test;
        for( unsigned long i = 0 ; i < n ; ++i ){
            if( i >= start && i < end ){
                x_test.push_back( x[i] );
            }else{
                x_train.push_back( x[i] );
            }
        }
        LinearAlgebra::Vector y_train( x_train.size() );
        LinearAlgebra::Vector y_test( x_test.size() );
        unsigned long train_idx = 0, test_idx = 0;
        for( unsigned long i = 0 ; i < n ; ++i ){
            if( i >= start && i < end ){
                y_test[test_idx++] = y[i];
            }else{
                y_train[train_idx++] = y[i];
            }
        }
        LinearAlgebra::Matrix k_train = Regression::kernelMatrix( x_train, f_kernel );
        LinearAlgebra::Matrix k_test = Regression::kernelMatrix( x_test, x_train, f_kernel );
        for( unsigned long i = 0 ; i < parameters.size() ; ++i ){
            double param = parameters[i];
            const Model<LinearAlgebra::Matrix>& model = f_train( k_train, y_train, param );
            LinearAlgebra::Vector y_hat = model.predict( k_test );
            errors[i] += f_error( y_test, y_hat ) / (double) kfold;
        }
    }

    //find best parameter and return it
    double best = 0;
    double min_error = INFINITY;
    for( unsigned long i = 0 ; i < parameters.size() ; ++i ){
        if( errors[i] < min_error ){
            min_error = errors[i];
            best = parameters[i];
        }
    }

    CrossValidation xval;
    xval.errors = errors;
    xval.parameters = parameters;
    xval.selected = best;

    return xval;

}

/* TODO
template< class TrainFunction, class ErrorFunction >
double crossValidate( LinearAlgebra::Matrix& x, LinearAlgebra::Vector& y, std::vector<double> parameters, const TrainFunction& f_train, const ErrorFunction& f_error, unsigned long kfold, bool x_is_kernel = false ){

        unsigned long n = y.size();

        //randomly permute the x matrix and y vector
        Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm( n );
        perm.setIdentity();
        std::random_shuffle( perm.indices().data(), perm.indices().data() + perm.indices().size() );

        if( x_is_kernel ){
            x = perm.transpose() * x * perm;
        }else{
            x = perm.transpose() * x;
        }
        y = perm.transpose() * y;

        double best = 0;
        double best_score = INFINITY;
        //perform cross validation
        for( unsigned long i = 0 ; i < parameters.size() ; ++i ){
            double param = parameters[i];
            double error = 0;
            for( unsigned long j = 0 ; j < kfold ; ++j ){
                unsigned long start = j * n / kfold;
                unsigned long length = ( j + 1 ) * n / kfold - start;
                unsigned long colstart, collength;
                if( x_is_kernel ){
                    colstart = start;
                    collength = length;
                }else{
                    colstart = 0;
                    collength = x.cols();
                }
                const Model<LinearAlgebra::Matrix>& model = f_train( x.block( start, colstart, length, collength ), y.segment( start, length ), param );
            }
        }

        //unpermute x and y
        perm = perm.inverse();
        if( x_is_kernel ){
            x = perm.transpose() * x * perm;
        }else{
            x = perm.transpose() * x;
        }
        y = perm.transpose() * y;


}
*/


#endif
