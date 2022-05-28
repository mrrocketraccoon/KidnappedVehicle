/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if(is_initialized == true)
  {
    return;
  }
  
  num_particles = 50;  // TODO: Set the number of particles
  std::default_random_engine gen;
  double std_x = std[0]; 
  double std_y = std[1]; 
  double std_theta = std[2];
  
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);

  for(uint index = 0; index < num_particles; index++)
  {
    Particle initial_particle;
    initial_particle.id = index;
    initial_particle.x = dist_x(gen);
    initial_particle.y = dist_y(gen);
    initial_particle.theta = dist_theta(gen);
    initial_particle.weight = 1.0;
    particles.push_back(initial_particle);    
    //std::cout << "At least I'm initializing particle partcile with id:" << initial_particle.id << " x:" << initial_particle.x << " y:" << initial_particle.y << " w:" << initial_particle.weight <<  std::endl;

  }
  is_initialized = true;  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double std_x = std_pos[0]; 
  double std_y = std_pos[1]; 
  double std_theta = std_pos[2];
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_x);
  std::normal_distribution<double> dist_y(0, std_y);
  std::normal_distribution<double> dist_theta(0, std_theta);
  //std::cout << "At least I'm entering in the predict function" << std::endl;
  for(auto& particle:particles)
  {
    double x_f = particle.x;
    double y_f = particle.y;
    double theta_f;

    if (fabs(yaw_rate) < 0.00001)
    {
      x_f += velocity * delta_t * cos(particle.theta);
      y_f += velocity * delta_t * sin(particle.theta);
      theta_f = particle.theta;
    }
    else
    {
      x_f = particle.x + (velocity/yaw_rate)*(sin(particle.theta+yaw_rate*delta_t)-sin(particle.theta));
      y_f = particle.y + (velocity/yaw_rate)*(cos(particle.theta)-cos(particle.theta+yaw_rate*delta_t));
      theta_f = particle.theta + yaw_rate*delta_t;
    }
    particle.x = x_f + dist_x(gen);
    particle.y = y_f + dist_y(gen);
    particle.theta = theta_f + dist_theta(gen);
    //std::cout << "At least I'm trying to predict partcile with id:" << particle.id << " x:" << particle.x << " y:" << particle.y << " w:" << particle.weight <<  std::endl;
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  LandmarkObs nearest_neighbor;

  for(auto& observation:observations)
  {  
    //initialize closest distance variable
    double closest_distance = std::numeric_limits<double>::max();
    for(auto& prediction:predicted)
    {  
      double distance = sqrt(pow(observation.x-prediction.x,2)+pow(observation.y-prediction.y,2));
      if(distance < closest_distance)
      {
        closest_distance = distance;
        nearest_neighbor = prediction;
      }
    }
    observation.id = nearest_neighbor.id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double std_landmark_range = std_landmark[0];
  double std_landmark_heading = std_landmark[1];

  for (auto& particle:particles)
  {
    // Find landmarks in particle's range.
    vector<LandmarkObs> in_range_landmarks;
    
    for(auto& landmark:map_landmarks.landmark_list)
    {
      float landmark_x = landmark.x_f;
      float landmark_y = landmark.y_f;
      int id = landmark.id_i;
      double dX = particle.x - landmark_x;
      double dY = particle.y - landmark_y;
      if (sensor_range >= sqrt(dX*dX + dY*dY))
      {
        in_range_landmarks.push_back(LandmarkObs{ id, landmark_x, landmark_y });
      }
    }
    
    // Transform observation coordinates from vehicle to map coordinate system.
    vector<LandmarkObs> map_coordinate_observations;
    for(auto& observation:observations)
    {
      double x_map = cos(particle.theta)*observation.x - sin(particle.theta)*observation.y + particle.x;
      double y_map = sin(particle.theta)*observation.x + cos(particle.theta)*observation.y + particle.y;
      map_coordinate_observations.push_back(LandmarkObs{ observation.id, x_map, y_map });
    }
    // Observation association to landmark.
    dataAssociation(in_range_landmarks, map_coordinate_observations);

    // Reseting weight.
    particle.weight = 1.0;
    // Calculate weights.
    for(auto& map_coordinate_observation:map_coordinate_observations)
    {
      double observation_x = map_coordinate_observation.x;
      double observation_y = map_coordinate_observation.y;
      int landmark_id = map_coordinate_observation.id;

      double landmark_x;
      double landmark_y;
      unsigned int k = 0;
      unsigned int n_landmarks = in_range_landmarks.size();
      bool found = false;
      while(!found && k < n_landmarks)
      {
        if(in_range_landmarks[k].id == landmark_id)
        {
          found = true;
          landmark_x = in_range_landmarks[k].x;
          landmark_y = in_range_landmarks[k].y;
        }
        k++;
      }
      // Calculating weight.
      double dX = observation_x - landmark_x;
      double dY = observation_y - landmark_y;

      double weight = (1/(2*M_PI*std_landmark_range*std_landmark_heading)) * exp( -( dX*dX/(2*std_landmark_range*std_landmark_range) + (dY*dY/(2*std_landmark_heading*std_landmark_heading))));

      if(weight == 0)
      {
        particle.weight *= 0.00001;
      }
      else
      {
        particle.weight *= weight;
      }
    }
    //std::cout << "At least I'm trying to update particle weights with id:" << particle.id << " x:" << particle.x << " y:" << particle.y << " w:" << particle.weight <<  std::endl;

  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  //std::cout << "At least I'm entering in the resample function" << std::endl;

  // Get weights and max weight.
  vector<double> weights;
  double maxWeight = std::numeric_limits<double>::min();
  for(int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if ( particles[i].weight > maxWeight ) {
      maxWeight = particles[i].weight;
    }
  }

  // Creating distributions.
  std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
  std::uniform_int_distribution<int> distInt(0, num_particles - 1);
  std::default_random_engine gen;

  // Generating index.
  int index = distInt(gen);

  double beta = 0.0;

  // the wheel
  vector<Particle> resampledParticles;
  for(int i = 0; i < num_particles; i++) {
    beta += distDouble(gen) * 2.0;
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }

  particles = resampledParticles;
  // for (auto& particle:particles)
  // {
  //   std::cout << "At least I'm trying to resample particles with id:" << particle.id << " x:" << particle.x << " y:" << particle.y << " w:" << particle.weight <<  std::endl;
  // }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}