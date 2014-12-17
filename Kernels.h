#ifndef KERNELS_INCLUDED__H
#define KERNELS_INCLUDED__H

#include<string>
#include<algorithm>

template< class S >
inline double positionalKmerKernel( const S& s0, const S& s1 ){

    unsigned long n = std::min( s0.size(), s1.size() );
    unsigned long val = 0, prev = 0, i = 0;
    for( ; i < n ; ++i ){
        if( s0[i] == s1[i] ){
            val += (++prev);
        }else{
            prev = 0;
        }
    }

    return val;

}

template< class Iter >
inline double positionalKmerKernel( Iter begin0, Iter end0, Iter begin1, Iter end1 ){

    unsigned long val = 0, prev = 0;
    while( begin0 != end0 && begin1 != end1 ){
        if( *begin0 == *begin1 ){
            val += (++prev);
        }else{
            prev = 0;
        }
        ++begin0; ++begin1;
    }

    return val;

}

#endif
