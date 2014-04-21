class pva{
    public:
        pva(float,float);
        pva ();

};

class pva_vector{
public:
    pva* pva_history;
    int num_pva;                                                                                      
    pva_vector();
    ~pva_vector();
    void add_pva(pva);
};    

class player_pva{                                                                                     
public:
    pva_vector position;                                                                              
    pva_vector velocity;
    pva_vector acceleration;                                                                          
    player_pva();
    void print_pos_history();
    void new_pos(float, float);                                                                       
    float predict_next(int);                                                                          
};  


