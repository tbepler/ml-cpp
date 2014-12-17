#include"LinearAlgebra.h"
#include"Regression.h"
#include"Model.h"
#include"Kernels.h"
#include"CrossValidation.h"
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<getopt.h>
#include<algorithm>
#include<cmath>

using namespace std;
using namespace LinearAlgebra;
using namespace Regression;

typedef double(*string_kernel)(const string&, const string&);
typedef KernelizedModel< vector< string >, string_kernel > string_model; 

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

ostream& operator<< ( ostream& out, const string_model& model ){
    vector<string> strs = model.data;
    Vector alphas = model.weights;
    double bias = model.bias;
    
    out << "Bias = " << bias << endl;
    out << "String\tAlpha" << endl;
    for( unsigned long i = 0 ; i < strs.size() ; ++i ){
        out << strs[i] << "\t" << alphas[i] << endl;
    }
    return out;
}

istream& operator>> ( istream& in, string_model& model ){
    //first read the bias
    double bias;
    //discard up to equals
    in.ignore( numeric_limits<streamsize>::max(), '=' );
    in >> bias;
    //discard remainder of bias line and header line
    string discard;
    getline( in, discard );
    getline( in, discard );

    //read strings and alphas using data struct
    Data data;
    in >> data;

    model.bias = bias;
    model.data = data.strs;
    model.weights = data.y;
    cerr << model.weights << endl;
    //set kernel to be pkk kernel -- TODO read and write kernel to file
    model.f_kernel = positionalKmerKernel<string>;

    return in;
}

struct TrainOpts{

    vector< string > strs;
    Vector y;
    unsigned long kfold = 5;
    vector< double > lambdas { .00001, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000 };
    ofstream model_out;
    ofstream pred_out;
    ofstream report_out;

};

void trainUsage( ostream& out ){
    out << "Usage: pkkridge train [-h] [-d FILE] [-l DOUBLE [DOUBLES...]] [-k INT] [-m FILE] [-p FILE] [-r FILE]" << endl;
    //TODO
}

bool parseTrainOpts( int argc, char* argv[], TrainOpts& opts ){
    static struct option long_opts[] = {
        { "help", no_argument, 0, 'h' },
        { "data", required_argument, 0, 'd' },
        { "lambdas", required_argument, 0, 'l' },
        { "kfold", required_argument, 0, 'k' },
        { "model", required_argument, 0, 'm' },
        { "predictions", required_argument, 0, 'p' },
        { "report", required_argument, 0, 'r' },
        { 0, 0, 0, 0 }
    };
    ifstream data_in;
    int opt_idx = 0;
    int c;
    while( ( c = getopt_long( argc, argv, "hd:l:k:m:p:r:", long_opts, &opt_idx ) ) != -1 ){

        switch( c ){
            case 'h':
                trainUsage( cerr );
                return false;
            case 'd':
                data_in.open( optarg );
                if( !data_in.is_open() ){
                    cerr << "Could not open data file: " << optarg << endl;
                    return false;
                }
                break;
            case 'l':
                //TODO
                opts.lambdas.clear();
                opts.lambdas.push_back( atof( optarg ) );
                //get lambdas up to next arg or end of args
                while( optind < argc && *argv[optind] != '-' ){
                    opts.lambdas.push_back( atof( argv[optind++] ) );
                }
                break;
            case 'k':
                opts.kfold = atoi( optarg );
                break;
            case 'm':
                opts.model_out.open( optarg );
                if( !opts.model_out.is_open() ){
                    cerr << "Could not open model file: " << optarg << endl;
                    return false;
                }
                break;
            case 'p':
                opts.pred_out.open( optarg );
                if( !opts.pred_out.is_open() ){
                    cerr << "Could not open prediction file: " << optarg << endl;
                    return false;
                }
                break;
            case 'r':
                opts.report_out.open( optarg );
                if( !opts.report_out.is_open() ){
                    cerr << "Could not open report file: " << optarg << endl;
                    return false;
                }
                break;
        }
    }

    istream& in = data_in.is_open() ? data_in : cin;
    Data data;
    in >> data;

    if( data_in.is_open() ){
        data_in.close();
    }

    opts.strs = data.strs;
    opts.y = data.y;
    return true;

}

/*
Options:
--help
--data
--kfold
--lambdas
--model
--predictions
--report
*/
int train( int argc, char* argv[] ){
    TrainOpts opts;
    if( !parseTrainOpts( argc, argv, opts ) ){
        return 1;
    }

    vector<string> strs = opts.strs;
    Vector y = opts.y;
    vector<double> lambdas = opts.lambdas;
    unsigned long kfold = opts.kfold;

    ostream& model_out = opts.model_out.is_open() ? opts.model_out : cout;
    ostream& report_out = opts.report_out.is_open() ? opts.report_out : cerr;

    double(*f_kernel)(const string&,const string&) = positionalKmerKernel<string>;

    double lambda;
    if( lambdas.size() > 1 ){
        //perform cross validation
        report_out << kfold << "-fold cross validation{" << endl;
    
        LinearModel(*f_train)( Matrix&, const Vector&, double ) = ridge;
        CrossValidation xval = crossValidate( strs, y, lambdas, f_kernel, f_train, meanSquaredError<Vector>, kfold ); 
        
        lambda = xval.selected;
        report_out << "Lambda\tError" << endl;
        for( unsigned long i = 0 ; i < lambdas.size() ; ++i ){
            report_out << xval.parameters[i] << "\t" << xval.errors[i] << endl;
        }
        report_out << "}" << endl;
    }else{
        lambda = lambdas[0];
    }

    report_out << "Training{" << endl;
    report_out << "Lambda = " << lambda << endl;
    const string_model& model = ridge( strs, y, lambda, f_kernel );
    Vector y_hat = model.predict( strs );
    if( opts.pred_out.is_open() ){
        opts.pred_out << "Actual\tPredicted" << endl;
        for( long i = 0 ; i < y.size() ; ++i ){
            opts.pred_out << y[i] << "\t" << y_hat[i] << endl;
        }
        opts.pred_out.close();
    }

    double mse = meanSquaredError( y, y_hat );
    double r = pearson( y, y_hat );
    double r2 = pow( r, 2 );
    report_out << "MSE = " << mse << endl;
    report_out << "r2 = " << r2 << endl;
    report_out << "}" << endl;

    model_out << model;

    return 0;

}

struct TestOpts{

    string_model model;
    Data data;
    ofstream report_out;
    ofstream pred_out;

};

void testUsage( ostream& out ){

    //TODO
    out << "Usage: pkkridge test [-h] [-m FILE] -d FILE [-r FILE] [-p FILE]" << endl;

}

bool parseTestOpts( int argc, char* argv[], TestOpts& opts ){

    static struct option long_opts[] = {
        { "help", no_argument, 0, 'h' },
        { "model", required_argument, 0, 'm' },
        { "data", required_argument, 0, 'd' },
        { "report", required_argument, 0, 'r' },
        { "predictions", required_argument, 0, 'p' },
        { 0, 0, 0, 0 }
    };

    ifstream model_in;
    ifstream data_in;
    int opt_idx = 0;
    int c;

    while( ( c = getopt_long( argc, argv, "hm:d:r:p:", long_opts, &opt_idx ) ) != -1 ){

        switch( c ){
            case 'h':
                testUsage( cerr );
                return false;
            case 'm':
                model_in.open( optarg );
                if( !model_in.is_open() ){
                    cerr << "Unable to open model file: " << optarg << endl;
                    return false;
                }
                break;
            case 'd':
                data_in.open( optarg );
                if( !data_in.is_open() ){
                    cerr << "Unable to open data file: " << optarg << endl;
                    return false;
                }
                break;
            case 'r':
                opts.report_out.open( optarg );
                if( !opts.report_out.is_open() ){
                    cerr << "Unable to open report file: " << optarg << endl;
                    return false;
                }
                break;
            case 'p':
                opts.pred_out.open( optarg );
                if( !opts.pred_out.is_open() ){
                    cerr << "Unable to open predictions file: " << optarg <<endl;
                    return false;
                }
                break;
        }

    }

    istream& m_in = model_in.is_open() ? model_in : cin;
    istream& d_in = data_in.is_open() ? data_in : cin;

    if( &m_in == &d_in ){
        cerr << "Error: at least one of `model' or `data' must be specified from a file" << endl;
        testUsage( cerr );
        return false;
    }

    string_model model;
    m_in >> model;

    Data data;
    d_in >> data;
    
    opts.model = model;
    opts.data = data;

    return true;

}

/*
Options:
--help
--model
--data
--report
--predictions
*/
int test( int argc, char* argv[] ){

    TestOpts opts;
    if( !parseTestOpts( argc, argv, opts ) ){
        return 1;
    }

    string_model model = opts.model;
    vector<string> strs = opts.data.strs;
    Vector y = opts.data.y;

    ostream& report_out = opts.report_out.is_open() ? opts.report_out : cout;
   
    report_out << "Testing{" << endl;
    Vector y_hat = model.predict( strs );
    if( opts.pred_out.is_open() ){
        opts.pred_out << "Actual\tPredicted" << endl;
        for( long i = 0 ; i < y.size() ; ++i ){
            opts.pred_out << y[i] << "\t" << y_hat[i] << endl;
        }
        opts.pred_out.close();
    }

    double mse = meanSquaredError( y, y_hat );
    double r = pearson( y, y_hat );
    double r2 = pow( r, 2 );
    report_out << "MSE = " << mse << endl;
    report_out << "r2 = " << r2 << endl;
    report_out << "}" << endl;

    return 0;

}

void usage( ostream& out ){
    out << "Usage: pkkridge [-h/--help] command [command-opts]" << endl;
    out << "Commands:" << endl << "train" << endl << "test" << endl;
}

int main( int argc, char* argv[] ){

    if( argc > 1 ){
        if( strcmp( argv[1], "train" ) == 0 ){
            return train( argc - 1, argv + 1 );
        }else if( strcmp( argv[1], "test" ) == 0 ){
            return test( argc - 1, argv + 1 );
        }else if( strcmp( argv[1], "-h" ) == 0 ){

        }else if( strcmp( argv[1], "--help" ) == 0 ){

        }else{
            cerr << "Unknown command: `" << argv[1] << "'" << endl;
        }
    }
    usage( cerr );
    return 0;

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
