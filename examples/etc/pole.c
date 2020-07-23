/*----------------------------------------------------------------------
    This file contains a simulation of the cart and pole dynamic system and
 a procedure for learning to balance the pole.  Both are described in
 Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
 Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
 Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983, and in Sutton, "Temporal
 Aspects of Credit Assignment in Reinforcement Learning", PhD
 Dissertation, Department of Computer and Information Science, University
 of Massachusetts, Amherst, 1984.  The following routines are included:

       main:              controls simulation interations and implements
                          the learning system.

       cart_and_pole:     the cart and pole dynamics; given action and
                          current state, estimates next state

       get_box:           The cart-pole's state space is divided into 162
                          boxes.  get_box returns the index of the box into
                          which the current state appears.

 These routines were written by Rich Sutton and Chuck Anderson.  Claude Sammut
 translated parts from Fortran to C.  Please address correspondence to
 sutton@gte.com or anderson@cs.colostate.edu
---------------------------------------
Changes:
  1/93: A bug was found and fixed in the state -> box mapping which resulted
        in array addressing outside the range of the array.  It's amazing this
        program worked at all before this bug was fixed.  -RSS
----------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define min(x, y)               ((x <= y) ? x : y)
#define max(x, y)	        ((x >= y) ? x : y)
#define prob_push_right(s)      (1.0 / (1.0 + exp(-max(-50.0, min(s, 50.0)))))
#define random                  ((float) rand() / (float)((1 << 31) - 1))

#define N_BOXES         162         /* Number of disjoint boxes of state space. */
#define ALPHA		1000        /* Learning rate for action weights, w. */
#define BETA		0.5         /* Learning rate for critic weights, v. */
#define GAMMA		0.95        /* Discount factor for critic. */
#define LAMBDAw		0.9         /* Decay rate for w eligibility trace. */
#define LAMBDAv		0.8         /* Decay rate for v eligibility trace. */

#define MAX_FAILURES     100         /* Termination criterion. */
#define MAX_STEPS        100000

typedef float vector[N_BOXES];

int
main()
{
  float x,			/* cart position, meters */
        x_dot,			/* cart velocity */
        theta,			/* pole angle, radians */
        theta_dot;		/* pole angular velocity */
  vector  w,			/* vector of action weights */
          v,			/* vector of critic weights */
          e,			/* vector of action weight eligibilities */
          xbar;			/* vector of critic weight eligibilities */
  float p, oldp, rhat, r;
  int box, i, y, steps = 0, failures=0, failed;

  printf("Seed? ");
  scanf("%d",&i);
  srand(i);

  /*--- Initialize action and heuristic critic weights and traces. ---*/
  for (i = 0; i < N_BOXES; i++)
    w[i] = v[i] = xbar[i] = e[i] = 0.0;

  /*--- Starting state is (0 0 0 0) ---*/
  x = x_dot = theta = theta_dot = 0.0;

  /*--- Find box in state space containing start state ---*/
  box = get_box(x, x_dot, theta, theta_dot);

  /*--- Iterate through the action-learn loop. ---*/
  while (steps++ < MAX_STEPS && failures < MAX_FAILURES)
    {
      /*--- Choose action randomly, biased by current weight. ---*/
      y = (random < prob_push_right(w[box]));

      /*--- Update traces. ---*/
      e[box] += (1.0 - LAMBDAw) * (y - 0.5);
      xbar[box] += (1.0 - LAMBDAv);

      /*--- Remember prediction of failure for current state ---*/
      oldp = v[box];

      /*--- Apply action to the simulated cart-pole ---*/
      cart_pole(y, &x, &x_dot, &theta, &theta_dot);

      /*--- Get box of state space containing the resulting state. ---*/
      box = get_box(x, x_dot, theta, theta_dot);

      if (box < 0)
	{
	  /*--- Failure occurred. ---*/
	  failed = 1;
	  failures++;
	  printf("Trial %d was %d steps.\n", failures, steps);
	  steps = 0;

	  /*--- Reset state to (0 0 0 0).  Find the box. ---*/
	  x = x_dot = theta = theta_dot = 0.0;
	  box = get_box(x, x_dot, theta, theta_dot);

	  /*--- Reinforcement upon failure is -1. Prediction of failure is 0. ---*/
	  r = -1.0;
	  p = 0.;
	}
      else
	{
 	  /*--- Not a failure. ---*/
	  failed = 0;

	  /*--- Reinforcement is 0. Prediction of failure given by v weight. ---*/
	  r = 0;
	  p= v[box];
	}

      /*--- Heuristic reinforcement is:   current reinforcement
	      + gamma * new failure prediction - previous failure prediction ---*/
      rhat = r + GAMMA * p - oldp;

      for (i = 0; i < N_BOXES; i++)
	{
	  /*--- Update all weights. ---*/
	  w[i] += ALPHA * rhat * e[i];
	  v[i] += BETA * rhat * xbar[i];
	  if (v[i] < -1.0)
	    v[i] = v[i];

	  if (failed)
	    {
	      /*--- If failure, zero all traces. ---*/
	      e[i] = 0.;
	      xbar[i] = 0.;
	    }
	  else
	    {
	      /*--- Otherwise, update (decay) the traces. ---*/
	      e[i] *= LAMBDAw;
	      xbar[i] *= LAMBDAv;
	    }
	}

    }
  if (failures == MAX_FAILURES)
    printf("Pole not balanced. Stopping after %d failures.",failures);
  else
    printf("Pole balanced successfully for at least %d steps\n", steps);
}


/*----------------------------------------------------------------------
   cart_pole:  Takes an action (0 or 1) and the current values of the
 four state variables and updates their values by estimating the state
 TAU seconds later.
----------------------------------------------------------------------*/

/*** Parameters for simulation ***/

#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5		  /* actually half the pole's length */
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0
#define TAU 0.02		  /* seconds between state updates */
#define FOURTHIRDS 1.3333333333333


cart_pole(action, x, x_dot, theta, theta_dot)
int action;
float *x, *x_dot, *theta, *theta_dot;
{
    float xacc,thetaacc,force,costheta,sintheta,temp;

    force = (action>0)? FORCE_MAG : -FORCE_MAG;
    costheta = cos(*theta);
    sintheta = sin(*theta);

    temp = (force + POLEMASS_LENGTH * *theta_dot * *theta_dot * sintheta)
		         / TOTAL_MASS;

    thetaacc = (GRAVITY * sintheta - costheta* temp)
	       / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
                                              / TOTAL_MASS));

    xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;

/*** Update the four state variables, using Euler's method. ***/

    *x  += TAU * *x_dot;
    *x_dot += TAU * xacc;
    *theta += TAU * *theta_dot;
    *theta_dot += TAU * thetaacc;
}

/*----------------------------------------------------------------------
   get_box:  Given the current state, returns a number from 1 to 162
  designating the region of the state space encompassing the current state.
  Returns a value of -1 if a failure state is encountered.
----------------------------------------------------------------------*/

#define one_degree 0.0174532	/* 2pi/360 */
#define six_degrees 0.1047192
#define twelve_degrees 0.2094384
#define fifty_degrees 0.87266

get_box(x,x_dot,theta,theta_dot)
float x,x_dot,theta,theta_dot;
{
  int box=0;

  if (x < -2.4 ||
      x > 2.4  ||
      theta < -twelve_degrees ||
      theta > twelve_degrees)          return(-1); /* to signal failure */

  if (x < -0.8)  		       box = 0;
  else if (x < 0.8)     	       box = 1;
  else		    	               box = 2;

  if (x_dot < -0.5) 		       ;
  else if (x_dot < 0.5)                box += 3;
  else 			               box += 6;

  if (theta < -six_degrees) 	       ;
  else if (theta < -one_degree)        box += 9;
  else if (theta < 0) 		       box += 18;
  else if (theta < one_degree) 	       box += 27;
  else if (theta < six_degrees)        box += 36;
  else	    			       box += 45;

  if (theta_dot < -fifty_degrees) 	;
  else if (theta_dot < fifty_degrees)  box += 54;
  else                                 box += 108;

  return(box);
}

/*----------------------------------------------------------------------
  Result of:  cc -o pole pole.c -lm          (assuming this file is pole.c)
              pole
----------------------------------------------------------------------*/
/*
Trial 1 was 21 steps.
Trial 2 was 12 steps.
Trial 3 was 28 steps.
Trial 4 was 44 steps.
Trial 5 was 15 steps.
Trial 6 was 9 steps.
Trial 7 was 10 steps.
Trial 8 was 16 steps.
Trial 9 was 59 steps.
Trial 10 was 25 steps.
Trial 11 was 86 steps.
Trial 12 was 118 steps.
Trial 13 was 218 steps.
Trial 14 was 290 steps.
Trial 15 was 19 steps.
Trial 16 was 180 steps.
Trial 17 was 109 steps.
Trial 18 was 38 steps.
Trial 19 was 13 steps.
Trial 20 was 144 steps.
Trial 21 was 41 steps.
Trial 22 was 323 steps.
Trial 23 was 172 steps.
Trial 24 was 33 steps.
Trial 25 was 1166 steps.
Trial 26 was 905 steps.
Trial 27 was 874 steps.
Trial 28 was 758 steps.
Trial 29 was 758 steps.
Trial 30 was 756 steps.
Trial 31 was 165 steps.
Trial 32 was 176 steps.
Trial 33 was 216 steps.
Trial 34 was 176 steps.
Trial 35 was 185 steps.
Trial 36 was 368 steps.
Trial 37 was 274 steps.
Trial 38 was 323 steps.
Trial 39 was 244 steps.
Trial 40 was 352 steps.
Trial 41 was 366 steps.
Trial 42 was 622 steps.
Trial 43 was 236 steps.
Trial 44 was 241 steps.
Trial 45 was 245 steps.
Trial 46 was 250 steps.
Trial 47 was 346 steps.
Trial 48 was 384 steps.
Trial 49 was 961 steps.
Trial 50 was 526 steps.
Trial 51 was 500 steps.
Trial 52 was 321 steps.
Trial 53 was 455 steps.
Trial 54 was 646 steps.
Trial 55 was 1579 steps.
Trial 56 was 1131 steps.
Trial 57 was 1055 steps.
Trial 58 was 967 steps.
Trial 59 was 1061 steps.
Trial 60 was 1009 steps.
Trial 61 was 1050 steps.
Trial 62 was 4815 steps.
Trial 63 was 863 steps.
Trial 64 was 9748 steps.
Trial 65 was 14073 steps.
Trial 66 was 9697 steps.
Trial 67 was 16815 steps.
Trial 68 was 21896 steps.
Trial 69 was 11566 steps.
Trial 70 was 22968 steps.
Trial 71 was 17811 steps.
Trial 72 was 11580 steps.
Trial 73 was 16805 steps.
Trial 74 was 16825 steps.
Trial 75 was 16872 steps.
Trial 76 was 16827 steps.
Trial 77 was 9777 steps.
Trial 78 was 19185 steps.
Trial 79 was 98799 steps.
Pole balanced successfully for at least 100001 steps
*/
