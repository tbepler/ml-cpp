#ifndef VIEW_INCLUDED__H
#define VIEW_INCLUDED__H

template< class T, class E >
class View{

    private:
        T& container;
        unsigned long start;
        unsigned long size;

    public:
        View( T& container, unsigned long start, unsigned long end ) :
            container( container ),
            start( start ),
            end( end )
            {};

        inline E operator[] ( unsigned long i ){
            return container[ start + i ];
        };

        inline unsigned long size( ) const{
            return size;
        }



}

#endif
