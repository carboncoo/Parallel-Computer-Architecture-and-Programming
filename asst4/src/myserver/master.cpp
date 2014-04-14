// Copyright 2013 Harry Q. Bovik (hbovik)

#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "server/messages.h"
#include "server/master.h"
#include "tools/work_queue.h"

#define CORES 2

struct Primes {
    int nums[4];
    int totalReq;
    int tag;
};

static struct Master_state {

  // The mstate struct collects all the master node state into one
  // place.  You do not need to preserve any of the fields below, they
  // exist only to implement the basic functionality of the starter
  // code.

  bool server_ready;
  int max_num_workers;
  int num_pending_client_requests;
  int num_current_workers;
  int num_queued_req;
  WorkQueue <Request_msg> workqueue;  
  Worker_handle *my_worker;
  std::vector <Request_msg> msgs;
  std::vector <Client_handle>waiting_client;
  std::map<std::string,std::string> job_to_response;
  int * jobs_per_worker;
  Client_handle client_handle;
  bool program_finished;
  int * empty_cycle;
  std::vector <Primes> prime;
} mstate;

int globaltag = 0;

void master_node_init(int max_workers, int& tick_period) {

  // set up tick handler to fire every 5 seconds. (feel free to
  // configure as you please)
  tick_period = 3;

  mstate.max_num_workers = max_workers;
  mstate.num_pending_client_requests = 0;
  mstate.num_current_workers = 0;
  mstate.num_queued_req = 0;
  // don't mark the server as ready until the server is ready to go.
  // This is actually when the first worker is up and running, not
  // when 'master_node_init' returnes
  mstate.server_ready = false;
  mstate.program_finished = false;
  // fire off a request for a new worker
  mstate.workqueue = WorkQueue<Request_msg>();;
  int tag = random();
  Request_msg req(tag);
  req.set_arg("name", "my worker 0");
  request_new_worker_node(req);

  // create array of worker_handles equal to maxSize
  mstate.my_worker = (Worker_handle *)malloc(sizeof(Worker_handle) * mstate.max_num_workers);
  mstate.jobs_per_worker = (int *)malloc(sizeof(int) * mstate.max_num_workers);
  mstate.empty_cycle = (int *)malloc(sizeof(int) * mstate.max_num_workers);
  for(int i = 0; i < mstate.max_num_workers; i++)
  {
    mstate.jobs_per_worker[i] = 0;
    mstate.empty_cycle[i] = 0;
  }
}

void handle_new_worker_online(Worker_handle worker_handle, int tag) {

  // 'tag' allows you to identify which worker request this response
  // corresponds to.  Since the starter code only sends off one new
  // worker request, we don't use it here.

  mstate.my_worker[mstate.num_current_workers] = worker_handle;
  mstate.num_current_workers++;
  // Now that a worker is booted, let the system know the server is
  // ready to begin handling client requests.  The test harness will
  // now start its timers and start hitting your server with requests.
  if (mstate.server_ready == false) {
    server_init_complete();
    mstate.server_ready = true;
  }
}

void handle_worker_response(Worker_handle worker_handle, const Response_msg& resp) {

  // Master node has received a response from one of its workers.
  // Here we directly return this response to the client.


  //send_client_response(mstate.waiting_client[mstate.num_current_workers - 1], resp);
  int tag = resp.get_tag();
  if(mstate.msgs.at(tag).get_arg("cmd").compare("compareprimes") != 0){
      send_client_response(mstate.waiting_client.at(tag), resp);
      mstate.num_pending_client_requests--;
      mstate.job_to_response[mstate.msgs.at(tag).get_request_string()] = resp.get_response();
      for(int i = 0; i < mstate.num_current_workers; i++)
      {
            if(mstate.my_worker[i] == worker_handle){
                mstate.jobs_per_worker[i]--;
                break;
            }
      }
  }
  else
  {
    printf("Got response from countprime %d %d\n", tag, atoi(resp.get_response().substr(0,1).c_str()));
    std::cout << resp.get_response() << std::endl; 
    for(int i = 0; i < mstate.prime.size(); i++)
    {
        if(mstate.prime.at(i).tag == tag){
            int num = atoi(resp.get_response().substr(0,1).c_str());
            mstate.prime.at(i).nums[num] = atoi(resp.get_response().substr(2, resp.get_response().size() - 2).c_str());
            mstate.prime.at(i).totalReq++;
            if(mstate.prime.at(i).totalReq == 4)
            {
               Response_msg dummy_resp(0);   
               if (mstate.prime.at(i).nums[1]-mstate.prime.at(i).nums[0] > mstate.prime.at(i).nums[3]-mstate.prime.at(i).nums[2]) 
                    dummy_resp.set_response("There are more primes in the first range.");
               else
                    dummy_resp.set_response("There are more primes in the second range.");
               send_client_response(mstate.waiting_client.at(tag), dummy_resp); 
               break; 
            }

        }
    } 

  }
  if(mstate.num_queued_req > 0)
  {
    mstate.num_queued_req--;
    Request_msg handle_newmsg = mstate.workqueue.get_work();   
  //  handle_newmsg.set_tag(globaltag);
    send_request_to_worker(worker_handle, handle_newmsg); 
    int jobs_for_worker = 0;
    int position = -1;
    for(int i = 0; i < mstate.num_current_workers; i++)
    {
        if(mstate.my_worker[i] == worker_handle){
            mstate.jobs_per_worker[i]++;
            jobs_for_worker = mstate.jobs_per_worker[i];
            position = i;
            break;
        }
    }
    if(mstate.num_queued_req > 0 && jobs_for_worker < CORES)
    {
         mstate.num_queued_req--;
        Request_msg handle_newmsg = mstate.workqueue.get_work();   
      //  handle_newmsg.set_tag(globaltag);
        send_request_to_worker(worker_handle, handle_newmsg); 
        mstate.jobs_per_worker[position]++;
    }

  }
  return; 
}

void handle_client_request(Client_handle client_handle, const Request_msg& client_req) {

  if(!(mstate.job_to_response.find(client_req.get_request_string()) == mstate.job_to_response.end()))
  {
    Response_msg resp(0);
    std::string temp = mstate.job_to_response[client_req.get_request_string()];
    resp.set_response(temp);
    send_client_response(client_handle, resp);
    return;
  }

  mstate.num_pending_client_requests++;
  // Check to see if the result has already been cached
  /*if(mstate.job_to_response.find(client_req) == mstate.job_to_response.end()) {
    mstate.job_to_response.find(client_req)->second; 
    return;    

  }*/
  // You can assume that traces end with this special message.  It
  // exists because it might be useful for debugging to dump
  // information about the entire run here: statistics, etc.
  if (client_req.get_arg("cmd") == "lastrequest") {
    mstate.program_finished= true;
    mstate.client_handle = client_handle; 
    return;
  }
  if (client_req.get_arg("cmd") == "compareprimes") {
        //std::cout << client_req.get_request_string() << std::endl;
        

        Request_msg r1(globaltag);
        Request_msg r2(globaltag);
        Request_msg r3(globaltag);
        Request_msg r4(globaltag);
        r1.set_arg("n", client_req.get_arg("n1").c_str());
        r2.set_arg("n", client_req.get_arg("n2").c_str());
        r3.set_arg("n", client_req.get_arg("n3").c_str());
        r4.set_arg("n", client_req.get_arg("n4").c_str());
        r1.set_arg("num", "0");    
        r2.set_arg("num", "1");    
        r3.set_arg("num", "2");    
        r4.set_arg("num", "3");    

 
        r1.set_arg("cmd", "countprimes");
        r2.set_arg("cmd", "countprimes");
        r3.set_arg("cmd", "countprimes");
        r4.set_arg("cmd", "countprimes");
        mstate.msgs.insert(mstate.msgs.end(), client_req);
        mstate.msgs.insert(mstate.msgs.end(), r1);
        mstate.msgs.insert(mstate.msgs.end(), r2);
        mstate.msgs.insert(mstate.msgs.end(), r3);
        mstate.msgs.insert(mstate.msgs.end(), r4);
        globaltag = globaltag + 5;
        bool first = false;
        bool second = false;
        bool third = false;
        for(int i = 0; i < mstate.num_current_workers; i++)
        {
            if(mstate.jobs_per_worker[i] < CORES)
            {
                printf("tag is %d \n",  globaltag - 5);
                std::cout<<client_req.get_request_string()<<std::endl;
                send_request_to_worker(mstate.my_worker[i], r1);
                mstate.jobs_per_worker[i]++; 
                first = true;
            }
            else if(mstate.jobs_per_worker[i] < CORES && first)
            {
                printf("tag is %d \n",  globaltag - 5);
                 send_request_to_worker(mstate.my_worker[i], r2);
                mstate.jobs_per_worker[i]++; 
                 second = true; 
            }
            else if(mstate.jobs_per_worker[i] < CORES && first && second)
            {
                printf("tag is %d \n",  globaltag - 5);
                 send_request_to_worker(mstate.my_worker[i], r3);
                mstate.jobs_per_worker[i]++; 
                 third = true; 
            }
            else if(mstate.jobs_per_worker[i] < CORES && first && second && third)
            {
                printf("tag is %d \n",  globaltag - 5);
                 send_request_to_worker(mstate.my_worker[i], r4);
                mstate.jobs_per_worker[i]++; 
            }
        }
      //  request_new_worker_node(newmsg) if there are no available workers to create left
        if(!first)
        {
            mstate.workqueue.put_work(r1);
            mstate.workqueue.put_work(r2); 
            mstate.workqueue.put_work(r3); 
            mstate.workqueue.put_work(r4); 
        }
        else if(!second)
        {
            mstate.workqueue.put_work(r2); 
            mstate.workqueue.put_work(r3); 
            mstate.workqueue.put_work(r4);     
        }
        else if(!third)
        {
            mstate.workqueue.put_work(r3); 
            mstate.workqueue.put_work(r4);     
        }
        else
        {
            mstate.workqueue.put_work(r4);     
        }




        return;
  }

  // The provided starter code cannot handle multiple pending client
  // requests.  The server returns an error message, and the checker
  // will mark the response as "incorrect"
  if (mstate.num_pending_client_requests > 0) {
  


    //pthread_t workers[MAX_THREADS];
    mstate.waiting_client.insert(mstate.waiting_client.end(), client_handle);
    mstate.msgs.insert(mstate.msgs.end(), client_req);
    bool free_jobs = false;
    bool handled_request = false; 
    for(int i = 0; i < mstate.num_current_workers; i++)
    {
        if(mstate.jobs_per_worker[i] < CORES && !handled_request)
        {
            free_jobs = true;
            std::cout<<client_req.get_request_string()<<std::endl;
            Request_msg worker_req(globaltag, client_req);
            send_request_to_worker(mstate.my_worker[i], worker_req);
            mstate.jobs_per_worker[i]++; 
            globaltag++;
            handled_request = true;
            return;  
        } 
        /*else if(mstate.jobs_per_worker[i] < CORES && handled_request)
        {
           if(mstate.num_queued_req > 0)
            {
                printf("Just dequeued, %d left\n\n", mstate.num_queued_req);
                mstate.num_queued_req--;
                Request_msg handle_newmsg = mstate.workqueue.get_work();   
              //  handle_newmsg.set_tag(globaltag);
                send_request_to_worker(mstate.my_worker[i], handle_newmsg);
            }
        }*/
    }
  //  request_new_worker_node(newmsg) if there are no available workers to create left
    if(free_jobs == false)
    {
           Request_msg queue_req(globaltag, client_req);
           mstate.workqueue.put_work(queue_req);
           mstate.num_queued_req++;
           globaltag++; 
           return;
    } 
  }

  // Save off the handle to the client that is expecting a response.
  // The master needs to do this it can response to this client later
  // when 'handle_worker_response' is called.
 // Fire off request to the worker.  Eventually the worker will
  // respond, and your 'handle_worker_response' event handler will be
  // called to forward the worker's response back to the server.
  //Request_msg worker_req(globaltag, client_req);
  //send_request_to_worker(mstate.my_worker[mstate.num_current_workers - 1], worker_req);
  //mstate.jobs_per_worker[mstate.num_current_workers - 1]++;
 // globaltag++;
  //mstate.waiting_client.insert(mstate.waiting_client.end(), client_handle);
  //mstate.msgs.insert(mstate.msgs.end(), client_req);
 
  // We're done!  This event handler now returns, and the master
  // process calls another one of your handlers when action is
  // required.

}


void handle_tick() {

  // TODO: you may wish to take action here.  This method is called at
  // fixed time intervals, according to how you set 'tick_period' in
  // 'master_node_init'.

  bool all_working = true;
    for(int i = 0; i < mstate.num_current_workers; i++)
    {
        if(mstate.jobs_per_worker[i] == 0 && mstate.num_current_workers > 1)
        {
            mstate.empty_cycle[i]++;
            if(mstate.empty_cycle[i] == 5){
                kill_worker_node(mstate.my_worker[i]);
                mstate.num_current_workers--;
                all_working = false;
                for(int j = i; j < mstate.num_current_workers; j++)
                {
                    mstate.my_worker[j] = mstate.my_worker[j+1];
                    mstate.jobs_per_worker[j] = mstate.jobs_per_worker[j+1];
                    mstate.empty_cycle[i] = mstate.empty_cycle[i+1];
                }
                i--;
            }
        }
        else if(mstate.jobs_per_worker[i] < CORES)
        {
            all_working = false;
            mstate.empty_cycle[i] = 0;
        }
        else{
            mstate.empty_cycle[i] = 0;
        }
    }

    if(all_working && mstate.max_num_workers > mstate.num_current_workers)
    {
       
       Request_msg req(random());
       //char str[10];
        //sprintf(str, "%d", mstate.num_current_workers);
       req.set_arg("name", "new guy");
       request_new_worker_node(req);
    }
    if(mstate.program_finished && mstate.num_pending_client_requests == 1 && mstate.num_queued_req == 0)
    {
        Response_msg resp(0);
        resp.set_response("ack");
        send_client_response(mstate.client_handle, resp);
 
    }
}

