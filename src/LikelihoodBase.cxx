/*
 * Copyright (c) 2009--2018, the KLFitter developer team
 *
 * This file is part of KLFitter.
 *
 * KLFitter is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * KLFitter is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with KLFitter. If not, see <http://www.gnu.org/licenses/>.
 */

#include "KLFitter/LikelihoodBase.h"

#include <cmath>
#include <iostream>
#include <string>

#include "BAT/BCLog.h"
#include "BAT/BCParameter.h"
#include "KLFitter/DetectorBase.h"
#include "KLFitter/Permutations.h"
#include "KLFitter/PhysicsConstants.h"
#include "TRandom3.h"

// ---------------------------------------------------------
KLFitter::LikelihoodBase::LikelihoodBase(ParticleCollection** particles)
  : BCModel()
  , m_particles_permuted(particles)
  , m_permutations(0)
  , m_detector(0)
  , m_event_probability(std::vector<double>(0))
  , m_do_integrate(0)
  , m_fit_is_nan(false)
  , m_use_jet_mass(false)
  , m_TFs_are_good(true)
  , m_btag_method(kNotag) {
  BCLog::SetLogLevel(BCLog::nothing);
  MCMCSetRandomSeed(123456789);
}

// ---------------------------------------------------------
KLFitter::LikelihoodBase::~LikelihoodBase() {
  // Clear the parameters container in the BCModel to circumvent
  // BAT-internal memory leak (known issue in 0.9.4.1).
  ClearParameters(true);
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetPhysicsConstants(KLFitter::PhysicsConstants* physicsconstants) {
  m_physics_constants = *physicsconstants;

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetInitialParameters(std::vector<double> const& parameters) {
  // check number of parameters
  if (static_cast<int>(parameters.size()) != NParameters()) {
    std::cout << "KLFitter::SetInitialPosition(). Length of vector does not equal the number of parameters." << std::endl;
    return 0;
  }

  // set starting point for MCMC
  MCMCSetInitialPositions(parameters);

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetInitialParametersNChains(std::vector<double> const& parameters, unsigned int nchains) {
  // check number of parameters
  if (static_cast<int>(parameters.size()) != NParameters()) {
    std::cout << "KLFitter::SetInitialPosition(). Length of vector does not equal the number of parameters." << std::endl;
    return 0;
  }

  // set starting point for MCMC
  std::vector< std::vector<double> > par(0.);

  for (unsigned int i = 0; i< nchains; ++i) {
    par.push_back(parameters);
  }

  MCMCSetInitialPositions(par);

  MCMCSetFlagInitialPosition(2);

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetParameterRange(int index, double parmin, double parmax) {
  // check index
  if (index < 0 || index >= NParameters()) {
    std::cout << " KLFitter::Combinatorics::SetParameterRange(). Index out of range." << std::endl;
    return 0;
  }

  // set parameter ranges in BAT
  GetParameter(index)->SetLowerLimit(parmin);
  GetParameter(index)->SetUpperLimit(parmax);

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetDetector(KLFitter::DetectorBase** detector) {
  // set pointer to pointer of detector
  m_detector = detector;

  if (*m_detector) RequestResolutionFunctions();

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetParticlesPermuted(KLFitter::ParticleCollection** particles) {
  // set pointer to pointer of permuted particles
  m_particles_permuted  = particles;

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetPermutations(std::unique_ptr<KLFitter::Permutations>* permutations) {
  // error code
  int err = 1;

  // set pointer to pointer of permutation object
  m_permutations = permutations;

  // return error code
  return err;
}

// ---------------------------------------------------------
double KLFitter::LikelihoodBase::ParMin(int index) {
  // check index
  if (index < 0 || index >= NParameters()) {
    std::cout << " KLFitter::Combinatorics::ParMin(). Index out of range." << std::endl;
    return 0;
  }

  // return parameter range from BAT
  return GetParameter(index)->GetLowerLimit();
}

// ---------------------------------------------------------
double KLFitter::LikelihoodBase::ParMax(int index) {
  // check index
  if (index < 0 || index >= NParameters()) {
    std::cout << " KLFitter::Combinatorics::ParMax(). Index out of range." << std::endl;
    return 0;
  }

  // return parameter range from BAT
  return GetParameter(index)->GetUpperLimit();
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::Initialize() {
  // error code
  int err = 1;

  // save the current permuted particles
  err *= SavePermutedParticles();

  // save the corresponding resolution functions
  err *= SaveResolutionFunctions();

  // adjust parameter ranges
  err *= AdjustParameterRanges();

  // set initial values
  // (only for Markov chains - initial parameters for other minimisation methods are set in Fitter.cxx)
  SetInitialParameters(GetInitialParameters());

  // return error code
  return err;
}

// ---------------------------------------------------------
double KLFitter::LikelihoodBase::LogEventProbability() {
  double logprob = 0;

  if (m_btag_method != kNotag) {
    double logprobbtag = LogEventProbabilityBTag();
    if (logprobbtag <= -1e99) return -1e99;
    logprob += logprobbtag;
  }

  // use integrated value of LogLikelihood (default)
  if (m_do_integrate) {
    logprob += log(GetIntegral());
  } else {
    logprob += LogLikelihood(GetBestFitParameters());
  }

  return logprob;
}

// ---------------------------------------------------------
double KLFitter::LikelihoodBase::LogEventProbabilityBTag() {
  double logprob = 0;

  double probbtag = 1;

  if (m_btag_method == kVeto) {
    // loop over all model particles.  calculate the overall b-tagging
    // probability which is the product of all probabilities.
    for (size_t i = 0; i < m_particles_model->partons.size(); ++i) {
      // get index of corresponding measured particle.
      int index = m_particles_model->partons.at(i).GetIdentifier();
      if (index < 0)
        continue;

      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(i).GetTrueFlavor();
      bool isBTagged = m_particles_model->partons.at(i).GetIsBTagged();
      if (trueFlavor == Particles::PartonTrueFlavor::kLight && isBTagged == true)
        probbtag = 0.;
    }

    if (probbtag > 0) {
      logprob += log(probbtag);
    } else {
      return -1e99;
    }
  } else if (m_btag_method == kVetoLight) {
    // loop over all model particles.  calculate the overall b-tagging
    // probability which is the product of all probabilities.
    for (size_t i = 0; i < m_particles_model->partons.size(); ++i) {
      // get index of corresponding measured particle.
      int index = m_particles_model->partons.at(i).GetIdentifier();
      if (index < 0)
        continue;

      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(i).GetTrueFlavor();
      bool isBTagged = m_particles_model->partons.at(i).GetIsBTagged();
      if (trueFlavor == Particles::PartonTrueFlavor::kB && isBTagged == false)
        probbtag = 0.;
    }

    if (probbtag > 0) {
      logprob += log(probbtag);
    } else {
      return -1e99;
    }
  } else if (m_btag_method == kVetoBoth) {
    // loop over all model particles.  calculate the overall b-tagging
    // probability which is the product of all probabilities.
    for (size_t i = 0; i < m_particles_model->partons.size(); ++i) {
      // get index of corresponding measured particle.
      int index = m_particles_model->partons.at(i).GetIdentifier();
      if (index < 0)
        continue;

      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(i).GetTrueFlavor();
      bool isBTagged = m_particles_model->partons.at(i).GetIsBTagged();
      if (trueFlavor == Particles::PartonTrueFlavor::kLight && isBTagged == true) {
        probbtag = 0.;
      } else if (trueFlavor == Particles::PartonTrueFlavor::kB && isBTagged == false) {
        probbtag = 0.;
      }
    }

    if (probbtag > 0) {
      logprob += log(probbtag);
    } else {
      return -1e99;
    }
  } else if (m_btag_method == kWorkingPoint) {
    for (size_t i = 0; i < m_particles_model->partons.size(); ++i) {
      // get index of corresponding measured particle.
      int index = m_particles_model->partons.at(i).GetIdentifier();
      if (index < 0)
        continue;

      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(i).GetTrueFlavor();
      bool isBTagged = m_particles_model->partons.at(i).GetIsBTagged();
      double efficiency = m_particles_model->partons.at(i).GetBTagEfficiency();
      double rejection = m_particles_model->partons.at(i).GetBTagRejection();
      if (rejection < 0 || efficiency < 0) {
        std::cout <<  " KLFitter::LikelihoodBase::LogEventProbability() : Your working points are not set properly! Returning 0 probability " << std::endl;
        return -1e99;
      }

      if (trueFlavor == Particles::PartonTrueFlavor::kLight && isBTagged) {
        logprob += log(1./rejection);
      } else if (trueFlavor == Particles::PartonTrueFlavor::kLight && !isBTagged) {
        logprob += log(1 - 1./rejection);
      } else if (trueFlavor == Particles::PartonTrueFlavor::kB && isBTagged) {
        logprob += log(efficiency);
      } else if (trueFlavor == Particles::PartonTrueFlavor::kB && !isBTagged) {
        logprob += log(1 - efficiency);
      } else {
        std::cout << " KLFitter::LikelihoodBase::LogEventProbability() : b-tagging association failed! " << std::endl;
      }
    }
  }

  return logprob;
}

// ---------------------------------------------------------
bool KLFitter::LikelihoodBase::NoTFProblem(std::vector<double> parameters) {
  m_TFs_are_good = true;
  this->LogLikelihood(parameters);
  return m_TFs_are_good;
}

// ---------------------------------------------------------
void KLFitter::LikelihoodBase::PropagateBTaggingInformation() {
  // get number of partons
  unsigned int npartons = m_particles_model->partons.size();

  // loop over all model particles.
  for (unsigned int i = 0; i < npartons; ++i) {
    // get index of corresponding measured particle.
    int index = m_particles_model->partons.at(i).GetIdentifier();

    if (index < 0) {
      continue;
    }

    m_particles_model->partons.at(index).SetIsBTagged((*m_particles_permuted)->partons.at(index).GetIsBTagged());
    m_particles_model->partons.at(index).SetBTagEfficiency((*m_particles_permuted)->partons.at(index).GetBTagEfficiency());
    m_particles_model->partons.at(index).SetBTagRejection((*m_particles_permuted)->partons.at(index).GetBTagRejection());
  }
}

// ---------------------------------------------------------.
std::vector <double> KLFitter::LikelihoodBase::GetBestFitParameters() {
  if (fCachedParameters.size() > 0) {
    return fCachedParameters;
  } else {
    return BCModel::GetBestFitParameters();
  }
}

// ---------------------------------------------------------.
std::vector <double> KLFitter::LikelihoodBase::GetBestFitParameterErrors() {
  if (fCachedParameterErrors.size() > 0) {
    return fCachedParameterErrors;
  } else {
    return BCModel::GetBestFitParameterErrors();
  }
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::RemoveForbiddenParticlePermutations() {
  // error code
  int err = 1;

  // only in b-tagging type kVetoNoFit
  if (!((m_btag_method == kVetoNoFit) || (m_btag_method == kVetoNoFitLight) || (m_btag_method == kVetoNoFitBoth || (m_btag_method == kVetoHybridNoFit))))
    return err;

  // remove all permutations where a b-tagged jet is in the position of a model light quark
  const KLFitter::ParticleCollection * particles = (*m_permutations)->Particles();
  int nPartons = particles->partons.size();

  // When using kVetoHybridNoFit, copy the Permutations object and try to run
  // with the kVetoNoFit option. If all permutations are vetoed, use the backup
  // Permutations object and run with the kVetoNoFitLight option.
  int nPartonsModel = m_particles_model->partons.size();
  if (m_btag_method == kVetoHybridNoFit) {
    KLFitter::Permutations permutationsCopy(**m_permutations);
    for (int iParton(0); iParton < nPartons; ++iParton) {
      bool isBtagged = particles->partons.at(iParton).GetIsBTagged();

      for (int iPartonModel(0); iPartonModel < nPartonsModel; ++iPartonModel) {
        Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(iPartonModel).GetTrueFlavor();
        if ((!isBtagged)||(trueFlavor != Particles::PartonTrueFlavor::kLight)) continue;
        err *= (*m_permutations)->RemoveParticlePermutations(Particles::Type::kParton, iParton, iPartonModel);
      }
    }

    if ((*m_permutations)->NPermutations() != 0) {
      return err;
    } else {
      **m_permutations = permutationsCopy;
    }
  }

  for (int iParton(0); iParton < nPartons; ++iParton) {
    bool isBtagged = particles->partons.at(iParton).GetIsBTagged();

    for (int iPartonModel(0); iPartonModel < nPartonsModel; ++iPartonModel) {
      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(iPartonModel).GetTrueFlavor();
      if ((m_btag_method == kVetoHybridNoFit)&&((isBtagged) || (trueFlavor != Particles::PartonTrueFlavor::kB)))
        continue;
      if ((m_btag_method == kVetoNoFit)&&((!isBtagged) || (trueFlavor != Particles::PartonTrueFlavor::kLight)))
        continue;
      if ((m_btag_method == kVetoNoFitLight)&&((isBtagged) || (trueFlavor != Particles::PartonTrueFlavor::kB)))
        continue;
      if ((m_btag_method == kVetoNoFitBoth)&&(((isBtagged)&&(trueFlavor != Particles::PartonTrueFlavor::kLight)) || ((!isBtagged)&&(trueFlavor != Particles::PartonTrueFlavor::kB))))
        continue;

      err *= (*m_permutations)->RemoveParticlePermutations(Particles::Type::kParton, iParton, iPartonModel);
    }
  }

  // return error code
  return err;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::SetParametersToCache(int iperm, int nperms) {
  // set correct size of cachevector
  if (iperm == 0) {
    fCachedParametersVector.clear();
    fCachedParametersVector.assign(nperms, std::vector<double>(NParameters(), 0));

    fCachedParameterErrorsVector.clear();
    fCachedParameterErrorsVector.assign(nperms, std::vector<double>(NParameters(), 0));

    fCachedNormalizationVector.clear();
    fCachedNormalizationVector.assign(nperms, 0.);
  }

  if ((iperm > static_cast<int>(fCachedParametersVector.size())) || (iperm > static_cast<int>(fCachedParameterErrorsVector.size()))) {
    std::cout << "KLFitter::LikelihoodBase::SetParametersToCache: iperm > size of fCachedParametersVector or fCachedParameterErrorsVector!" << std::endl;
    return 0;
  }
  fCachedParametersVector.at(iperm) = BCModel::GetBestFitParameters();
  fCachedParameterErrorsVector.at(iperm) = BCModel::GetBestFitParameterErrors();
  fCachedNormalizationVector.at(iperm) = BCIntegrate::GetIntegral();

  int switchpar1 = -1;
  int switchpar2 = -1;
  double switchcache = 0;
  int partner = LHInvariantPermutationPartner(iperm, nperms, &switchpar1, &switchpar2);

  if (partner > iperm) {
    if ((static_cast<int>(fCachedParametersVector.size()) > partner) && (static_cast<int>(fCachedParameterErrorsVector.size()) > partner)) {
      fCachedParametersVector.at(partner) = BCModel::GetBestFitParameters();
      switchcache = fCachedParametersVector.at(partner).at(switchpar1);
      fCachedParametersVector.at(partner).at(switchpar1) = fCachedParametersVector.at(partner).at(switchpar2);
      fCachedParametersVector.at(partner).at(switchpar2) = switchcache;

      fCachedParameterErrorsVector.at(partner) = BCModel::GetBestFitParameterErrors();
      switchcache = fCachedParameterErrorsVector.at(partner).at(switchpar1);
      fCachedParameterErrorsVector.at(partner).at(switchpar1) = fCachedParameterErrorsVector.at(partner).at(switchpar2);
      fCachedParameterErrorsVector.at(partner).at(switchpar2) = switchcache;

      fCachedNormalizationVector.at(partner) = BCIntegrate::GetIntegral();
    } else {
      std::cout << "KLFitter::LikelihoodBase::SetParametersToCache: size of fCachedParametersVector too small!" << std::endl;
    }
  }
  GetParametersFromCache(iperm);

  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodBase::GetParametersFromCache(int iperm) {
  if ((static_cast<int>(fCachedParametersVector.size()) > iperm) && (static_cast<int>(fCachedParameterErrorsVector.size()) > iperm)) {
    fCachedParameters = fCachedParametersVector.at(iperm);
    fCachedParameterErrors = fCachedParameterErrorsVector.at(iperm);
    fCachedNormalization = fCachedNormalizationVector.at(iperm);
  } else {
    std::cout << "KLFitter::LikelihoodBase::GetParametersFromCache: size of fCachedParametersVector,  fCachedParameterErrorsVector or fCachedNormalizationVector too small!" << std::endl;
  }
  return 1;
}

// ---------------------------------------------------------.
double KLFitter::LikelihoodBase::GetIntegral() {
  if (fCachedNormalizationVector.size() > 0) {
    return fCachedNormalization;
  } else {
    return BCIntegrate::GetIntegral();
  }
}

// ---------------------------------------------------------.
int KLFitter::LikelihoodBase::ResetCache() {
  fCachedParameters.clear();
  fCachedParameterErrors.clear();

  fCachedNormalization = 0.;

  return 1;
}

// ---------------------------------------------------------.
double KLFitter::LikelihoodBase::SetPartonMass(double jetmass, double quarkmass, double *px, double *py, double *pz, double e) {
  double mass(0.);
  if (m_use_jet_mass) {
    mass = jetmass > 0. ? jetmass : 0.;
  } else {
    mass = quarkmass;
  }
  double p_orig = sqrt(*px * *px + *py * *py + *pz * *pz);
  double p_newmass = sqrt(e * e - mass * mass);
  double scale = p_newmass / p_orig;
  *px *= scale;
  *py *= scale;
  *pz *= scale;
  return mass;
}

// ---------------------------------------------------------.
double KLFitter::LikelihoodBase::GetBestFitParameter(unsigned int index) {
  if (fCachedParameters.size() > 0) {
    return fCachedParameters.at(index);
  } else {
    return BCModel::GetBestFitParameter(index);
  }
}

// ---------------------------------------------------------.
double KLFitter::LikelihoodBase::GetBestFitParameterError(unsigned int index) {
  if (fCachedParameterErrors.size() > 0) {
    return fCachedParameterErrors.at(index);
  } else {
    return BCModel::GetBestFitParameterError(index);
  }
}
