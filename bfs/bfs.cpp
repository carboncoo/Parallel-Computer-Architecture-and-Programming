#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include "CycleTimer.h"
#include "bfs.h"
#include "graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define GROUP 1000                  //Reduce CAS on bottom_up_step

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->alloc_count = count;
	list->present = (int*)calloc(list->alloc_count, sizeof(int));
    vertex_set_clear(list);
}

void bottom_up_step_omp(
    graph* g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances,
	int step)
{
	int numNodes = g->num_nodes;
	
	#pragma omp parallel for
	for(int curNode=0;curNode<numNodes;curNode+=GROUP){
		int tmp[GROUP];
		int tmpIndex = 0;
		int stepCurNode;
		
		int incoming, index, start_edge, end_edge, parent;
		for(int i=0;i<GROUP;i++){
			stepCurNode = curNode+i;
			
			if(stepCurNode<numNodes && distances[stepCurNode] == NOT_VISITED_MARKER){
				start_edge = g->incoming_starts[stepCurNode];
				end_edge = (stepCurNode < g->num_nodes-1) ? g->incoming_starts[stepCurNode+1] : g->num_edges;
				
				for (parent=start_edge; parent<end_edge; parent++) {
					incoming = g->incoming_edges[parent];

					if (distances[incoming] == step-1) {
						distances[stepCurNode] = step;
						
						tmp[tmpIndex++] = stepCurNode;
						
						break;
					}
				}
			}
		}
		
		if(tmpIndex>0){
			do{
				index = new_frontier->count;
			} while(!__sync_bool_compare_and_swap(&new_frontier->count, index, index+tmpIndex));
			
			for(int i=0;i<tmpIndex;i++){
				new_frontier->present[index+i] = tmp[i];
			}
		}
	}
}

void top_down_step_omp(
    graph* g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances,
	int step)
{
	#pragma omp parallel for
	for (int i=0; i<frontier->count;i++) {

        int node = frontier->present[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[node+1];

		int outgoing, index;
		if(end_edge>start_edge){
			int tmp[end_edge - start_edge];
			int tmpIndex = 0;
		
			for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
				outgoing = g->outgoing_edges[neighbor];

				if (distances[outgoing] == NOT_VISITED_MARKER) {
					distances[outgoing] = step;
					
					tmp[tmpIndex++] = outgoing;
				}
			}
			
			if(tmpIndex>0){
				do{
					index = new_frontier->count;
				} while(!__sync_bool_compare_and_swap(&new_frontier->count, index, index+tmpIndex));
				
				for(int i=0;i<tmpIndex;i++){
					new_frontier->present[index+i] = tmp[i];
				}
			}
		}
    }
}

void bfs_hybrid(graph* graph, solution* sol){
	int totalNodes = graph->num_nodes;
	
	vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, totalNodes);
    vertex_set_init(&list2, totalNodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
	#pragma omp parallel for
    for (int i=0; i<totalNodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
	
	int step = 0;
	while (frontier->count != 0) {

        vertex_set_clear(new_frontier);
		step++;
		
		//10 is the threshold ratio between total nodes and frontier count, under it go to bottom-up method, otherwise go to top-down method
		if(frontier->count*10 < totalNodes){
			//the frontier size is small, use top-down method
			top_down_step_omp(
				graph, 
				frontier, 
				new_frontier, 
				sol->distances, 
				step);
		}else {
			//the frontier size of large, use bottom-up method
			bottom_up_step_omp(
				graph, 
				frontier, 
				new_frontier, 
				sol->distances, 
				step);
		}

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_bottom_up_omp(graph* graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
	}

    // setup frontier with the root node
    frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

	int step=0;
    while (frontier->count != 0) {

        vertex_set_clear(new_frontier);
		step++;
        bottom_up_step_omp(
			graph, 
			frontier, 
			new_frontier, 
			sol->distances, 
			step);

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_top_down_omp(graph* graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
	}

    // setup frontier with the root node
    frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

	int step = 0;
    while (frontier->count != 0) {

        vertex_set_clear(new_frontier);
		
		step++;
        top_down_step_omp(graph, frontier, new_frontier, sol->distances, step);

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}



// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    graph* g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{

    for (int i=0; i<frontier->count; i++) {

        int node = frontier->present[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[node+1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->present[index] = outgoing;
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(graph* graph, solution* sol) {
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->present[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
	
    while (frontier->count != 0) {

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
