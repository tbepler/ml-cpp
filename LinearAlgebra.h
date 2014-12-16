#ifndef MATRIX_INCLUDED__H
#define MATRIX_INCLUDED__H

#include<Eigen/Dense>
#include<iostream>
#include<string>
#include<vector>

namespace LinearAlgebra{

    typedef Eigen::MatrixXd Matrix;
    typedef Eigen::MatrixXd::RowXpr Row;

    template< typename M, typename R = typename M::RowXpr >
    class RowAccessor{
        private:
            M& matrix;
        public:
            RowAccessor( M& m ) : matrix(m) {};

            inline unsigned long size() const{ return matrix.rows(); };

            inline R operator[]( unsigned long i ) const{
                return matrix.row(i);
            };

            inline R operator[]( unsigned long i ){
                return matrix.row(i);
            };
    };

    template< typename M, typename R = typename M::RowXpr >
    inline RowAccessor<M,R> rows( M& matrix ){
        RowAccessor<M,R> access( matrix );
        return access;
    }

    template<typename X, typename A, typename B>
    inline X solve( const A& a, const B& b){
        return a.ldlt().solve( b );
        //return a.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve( b );
        //return (a.transpose() * a).ldlt().solve(a.transpose() * b);
    }

    template<typename X, typename A, typename B>
    inline X leastSquares( const A& a, const B& b){
        //return a.ldlt().solve( b );
        //return a.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve( b );
        return (a.transpose() * a).ldlt().solve(a.transpose() * b);
    }

    template<typename V>
    inline double dot( const V& v0, const V& v1 ){
        return v0.dot( v1 );
    }

    Matrix readMatrix( std::istream& in ){
        unsigned long rows = 0, cols = 0;
        std::vector<double> coefs;
        std::string line;
        //terminate on empty line or end of stream
        while( std::getline( in, line ) && !line.empty() ){
            std::stringstream ss( line );
            unsigned long temp_cols = 0;
            double coef;
            while( !ss.eof() ){
                ss >> coef;
                coefs.push_back( coef );
                ++temp_cols;
            }
            if( cols <= 0 ){
                cols = temp_cols;
            }else if( cols != temp_cols ){
                //the file contains inconsistent number of columns
                //throw an error
                //TODO
            }
            ++rows;
        }
        
        Matrix matrix( rows, cols );
        unsigned long i,j;
        for( i = 0 ; i < rows ; ++i ){
            for( j = 0 ; j < cols ; ++j ){
                matrix( i, j ) = coefs[ i * cols + j ];
            }
        }

        return matrix;
    }

    

}


#endif
