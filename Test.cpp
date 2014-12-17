#include"LinearAlgebra.h"
#include"Regression.h"
#include"Model.h"
#include"Kernels.h"
#include<iostream>
#include<fstream>
#include<Eigen/Dense>

using namespace std;
using namespace LinearAlgebra;
using namespace Regression;

struct Data{

    vector<string> strs;
    Vector y;

};

void readStrings( istream& in, Data& data ){

    vector<string> strs;
    vector<double> vals;
    string line;
    while( getline( in, line ) ){
        if( !line.empty() ){
            stringstream ss( line );
            string str;
            double val;
            ss >> str;
            ss >> val;
            strs.push_back( str );
            vals.push_back( val );
        }
    }

    Vector y( vals.size() );
    for( unsigned long i = 0 ; i < vals.size() ; ++i ){
        y[i] = vals[i];
    }

    data.strs = strs;
    data.y = y;

}

ostream& operator<< ( ostream& out, const Data& data ){
    vector<string> strs = data.strs;
    Vector y = data.y;
    for( unsigned long i = 0 ; i < strs.size() ; ++i ){
        out << strs[i] << " " << y[i] << endl;
    }
    return out;
}

istream& operator>> ( istream& in, Data& data ){
    readStrings( in, data );
    return in;
}

int main( int argc, char* argv[] ){

    fstream fin;
    fin.open( argv[1] );
    Data train;
    fin >> train;
    fin.close();

    cout << "Training data:" << endl;
    cout << train << endl;

    double(*f_kernel)(const string&,const string&) = positionalKmerKernel<string>;
    const Model< vector<string> >& model = ridge( train.strs, train.y, 1, f_kernel );

    Vector y_hat = model.predict( train.strs );
    double mse = meanSquaredError( train.y, y_hat );
    
    cout << "Training predictions:" << endl;
    cout << "String\tActual\tPredicted" << endl;
    for( int i = 0 ; i < y_hat.size() ; ++i ){
        cout << train.strs[i] << "\t" << train.y[i] << "\t" << y_hat[i] << endl;
    }
    cout << "MSE = " << mse << endl << endl;

    for( int i = 2 ; i < argc ; ++i ){
        Data test;
        fstream fin;
        fin.open( argv[i] );
        fin >> test;
        fin.close();

        y_hat = model.predict( test.strs );
        mse = meanSquaredError( test.y, y_hat );
    
        cout << argv[i] << " predictions:" << endl;
        cout << "String\tActual\tPredicted" << endl;
        for( int i = 0 ; i < y_hat.size() ; ++i ){
            cout << test.strs[i] << "\t" << test.y[i] << "\t" << y_hat[i] << endl;
        }
        cout << "MSE = " << mse << endl << endl;
    }


}
