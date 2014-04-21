/* Movement vector library for 18-551 project Spring 14
	Ajmal Thanikkal
	Dev Shah
	Gavriel Adler*/
#define MAX SECONDS*Ta //max pvas a movement vector can hold
#define SECONDS 100
#define Ta 100 //samples per second
#define TAN61 1.8040
#define VIDEO_COURT_W 770
#define VIDEO_COURT_H 385
#define NCAA_COURT_W 50
#define NCAA_COURT_H 47
#define SIDE_EMPTY_SPACE_W 42

#include <iostream>
#include <cmath>
using namespace std;

class pva;
class pva_vector;
class player_pva;

//single pva--position, velocity, or accelecration
class pva{
public:
	float x, y, absolute;
	pva (float, float);
	pva ();
};

pva::pva(){
	x = 0;
	y = 0;
	absolute = 0;
}

pva::pva(float new_x, float new_y){
    //convert using trapazoidal model
    float adjacent = new_y/TAN61;
    float width = VIDEO_COURT_W-(2*adjacent);
    x = (NCAA_COURT_W*(new_x-SIDE_EMPTY_SPACE_W))/width;
    y = (NCAA_COURT_H*new_y)/VIDEO_COURT_H;
    //adjust for inexact numbers
    if (x > NCAA_COURT_W) x = NCAA_COURT_W;
    if (x < 0) x = 0;
    if (y > NCAA_COURT_H) y = NCAA_COURT_H;
    if (y < 0) y = 0;
    absolute = sqrt(x*x+y*y);
}

//vector of pvas representing position
class pva_vector{
public:
	pva* pva_history;
	int num_pva;
	pva_vector();
	~pva_vector();
	void add_pva(pva);
};

pva_vector::pva_vector(){
	num_pva = 0;
	pva_history = new pva[MAX];
}

pva_vector::~pva_vector(){
	//delete[] pva_history;
}

void pva_vector::add_pva(pva new_pva){
	pva_history[num_pva] = new_pva;
	num_pva++;
}

//one player's movement info
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

player_pva::player_pva(){
	pva_vector position;
	pva_vector velocity;
	pva_vector acceleration;
}

void player_pva::print_pos_history(){
    int i;
    for (i=0; i < position.num_pva; i++){
        float this_x = position.pva_history[i].x;
        float this_y = position.pva_history[i].y;
        cout << i << ": " << this_x << ", " << this_y << "\n";
    }
}

//calculate and add new P/V/A data when new position data received
void player_pva::new_pos(float x, float y){
	//create new position date
	pva new_pos(x, y);
	position.add_pva(new_pos);

	//create new velocity data
	if (position.num_pva > 1){
		float old_posx = position.pva_history[position.num_pva-2].x;
		float new_velx = (new_pos.x - old_posx)/Ta;
		float old_posy = position.pva_history[position.num_pva-2].y;
		float new_vely = (new_pos.y - old_posy)/Ta;
		pva new_vel(new_velx, new_vely);
		velocity.add_pva(new_vel);
		//create new acceleration data
		if (velocity.num_pva > 1){
			float old_velx = velocity.pva_history[velocity.num_pva-2].x;
			float new_accelx = (new_vel.x - old_velx)/Ta;
			float old_vely = velocity.pva_history[velocity.num_pva-2].y;
			float new_accely = (new_vel.y - old_vely)/Ta;
			pva new_accel(new_accelx, new_accely);
			acceleration.add_pva(new_accel);
		}
	}

}

//predict a next point based on current trajectory
//x = 1 for x, x = 0 for y
float player_pva::predict_next(int x){
	float last_pos = 0;
	float last_vel = 0;
	float last_accel = 0;
	if (position.num_pva){
		pva last_pos_pva = position.pva_history[position.num_pva-1];
		if (x) last_pos = last_pos_pva.x;
		else last_pos = last_pos_pva.y;
	}
	if (velocity.num_pva){
		pva last_vel_pva = velocity.pva_history[velocity.num_pva-1];
		if (x) last_vel = last_vel_pva.x;
		else last_vel = last_vel_pva.y;
	}
	if (acceleration.num_pva){
		pva last_accel_pva = acceleration.pva_history[acceleration.num_pva-1];
		if (x) last_accel = last_accel_pva.x;
		else last_accel = last_accel_pva.y;
	}
	return last_pos + (last_vel + (last_accel*Ta))*Ta;
}
/*
int main(){
	player_pva lebron;
	lebron.new_pos(500, 150);
	lebron.new_pos(500, 180);
	lebron.new_pos(600, 200);
	lebron.new_pos(650, 150);
	lebron.new_pos(700, 100);
    lebron.print_pos_history();
	float guessx = lebron.predict_next(1);
	float guessy = lebron.predict_next(0);
	cout << guessx << "\n";
	cout << guessy << "\n";
	return 0;
}*/


