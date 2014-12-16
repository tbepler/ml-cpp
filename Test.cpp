#include"LinearAlgebra.h"
#include"Regression.h"
#include<iostream>
#include<fstream>
#include<Eigen/Dense>

using namespace std;
using namespace LinearAlgebra;
using namespace Regression;



int main( int argc, char* argv[] ){

    ifstream in;
    in.open( argv[1] );
    Matrix a = readMatrix( in );
    cout << "Matrix A:" << endl;
    cout << a << endl << endl;
    in.close();

    in.open( argv[2] );
    Matrix b = readMatrix( in );
    cout << "Matrix b:" << endl;
    cout << b << endl << endl;
    in.close();

    cout << "Matrix A^T * A:" << endl;
    cout << a.transpose() * a << endl << endl;

    Matrix x = leastSquares<Matrix>( a, b );
    cout << "Matrix x:" << endl;
    cout << x << endl << endl;

    cout << "Matrix A*x:" << endl;
    cout << a * x << endl << endl;

    double lambda;
    int i;
    for( i = 10 ; i >= 0 ; i -= 5 ){
        lambda = i;
        Matrix alpha = ridge( rows( a ), b, lambda, dot<Row> );
        cout << "Ridge alphas with lambda = " << lambda << ":" << endl;
        cout << alpha << endl << endl;
        cout << "Matrix K^T*alpha:" << endl;
        cout << kernelMatrix( rows( a ), dot<Row> ).transpose() * alpha << endl << endl;
    }


}
