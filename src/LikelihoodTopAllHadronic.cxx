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

#include "KLFitter/LikelihoodTopAllHadronic.h"

#include <algorithm>
#include <iostream>
#include <set>

#include "BAT/BCMath.h"
#include "BAT/BCParameter.h"
#include "KLFitter/DetectorBase.h"
#include "KLFitter/Particles.h"
#include "KLFitter/Permutations.h"
#include "KLFitter/PhysicsConstants.h"
#include "KLFitter/ResolutionBase.h"
#include "TLorentzVector.h"

// ---------------------------------------------------------
KLFitter::LikelihoodTopAllHadronic::LikelihoodTopAllHadronic()
  : KLFitter::LikelihoodBase::LikelihoodBase()
  , fFlagTopMassFixed(false)
  , fFlagGetParSigmasFromTFs(false) {
  // define model particles
  DefineModelParticles();

  // define parameters
  DefineParameters();
}

// ---------------------------------------------------------
KLFitter::LikelihoodTopAllHadronic::~LikelihoodTopAllHadronic() = default;

// ---------------------------------------------------------
int KLFitter::LikelihoodTopAllHadronic::DefineModelParticles() {
  // create the particles of the model
  fParticlesModel.reset(new KLFitter::Particles{});

  // add model particles
  // create dummy TLorentzVector
  TLorentzVector dummy{0, 0, 0, 0};  // 4-vector
  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,  // type
                               "hadronic b quark 1",          // name
                               0,                             // index of corresponding particle
                               KLFitter::Particles::kB);      // b jet (truth)

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,
                               "hadronic b quark 2",
                               1,                             // index of corresponding particle
                               KLFitter::Particles::kB);      // b jet (truth)

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,
                               "light quark 1",
                               2,                             // index of corresponding particle
                               KLFitter::Particles::kLight);  // light jet (truth)

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,
                               "light quark 2",
                               3,                             // index of corresponding particle
                               KLFitter::Particles::kLight);  // light jet (truth)

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,
                               "light quark 3",
                               4,                             // index of corresponding particle
                               KLFitter::Particles::kLight);  // light jet (truth)

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,
                               "light quark 4",
                               5,                             // index of corresponding particle
                               KLFitter::Particles::kLight);  // light jet (truth)

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kBoson,
                               "hadronic W 1");

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kBoson,
                               "hadronic W 2");

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,
                               "hadronic top 1");

  fParticlesModel->AddParticle(&dummy,
                               KLFitter::Particles::kParton,
                               "hadronic top 2");

  // no error
  return 1;
}

// ---------------------------------------------------------
void KLFitter::LikelihoodTopAllHadronic::DefineParameters() {
  // add parameters of model
  AddParameter("energy hadronic b 1",       fPhysicsConstants.MassBottom(), 1000.0);  // parBhad1E
  AddParameter("energy hadronic b 2",       fPhysicsConstants.MassBottom(), 1000.0);  // parBhad2E
  AddParameter("energy light quark 1",    0.0, 1000.0);                                // parLQ1E
  AddParameter("energy light quark 2",    0.0, 1000.0);                                // parLQ2E
  AddParameter("energy light quark 3",    0.0, 1000.0);                                // parLQ3E
  AddParameter("energy light quark 4",    0.0, 1000.0);                                // parLQ4E
  AddParameter("top mass",              100.0, 1000.0);                                // parTopM
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTopAllHadronic::CalculateLorentzVectors(std::vector <double> const& parameters) {
  static double scale;
  static double whad1_fit_e;
  static double whad1_fit_px;
  static double whad1_fit_py;
  static double whad1_fit_pz;
  static double whad2_fit_e;
  static double whad2_fit_px;
  static double whad2_fit_py;
  static double whad2_fit_pz;
  static double thad1_fit_e;
  static double thad1_fit_px;
  static double thad1_fit_py;
  static double thad1_fit_pz;
  static double thad2_fit_e;
  static double thad2_fit_px;
  static double thad2_fit_py;
  static double thad2_fit_pz;

  // hadronic b quark 1
  bhad1_fit_e = parameters[parBhad1E];
  scale = sqrt(bhad1_fit_e*bhad1_fit_e - bhad1_meas_m*bhad1_meas_m) / bhad1_meas_p;
  bhad1_fit_px = scale * bhad1_meas_px;
  bhad1_fit_py = scale * bhad1_meas_py;
  bhad1_fit_pz = scale * bhad1_meas_pz;

  // hadronic b quark 2
  bhad2_fit_e = parameters[parBhad2E];
  scale = sqrt(bhad2_fit_e*bhad2_fit_e - bhad2_meas_m*bhad2_meas_m) / bhad2_meas_p;
  bhad2_fit_px = scale * bhad2_meas_px;
  bhad2_fit_py = scale * bhad2_meas_py;
  bhad2_fit_pz = scale * bhad2_meas_pz;

  // light quark 1
  lq1_fit_e = parameters[parLQ1E];
  scale = sqrt(lq1_fit_e*lq1_fit_e - lq1_meas_m*lq1_meas_m) / lq1_meas_p;
  lq1_fit_px = scale * lq1_meas_px;
  lq1_fit_py = scale * lq1_meas_py;
  lq1_fit_pz = scale * lq1_meas_pz;

  // light quark 2
  lq2_fit_e = parameters[parLQ2E];
  scale = sqrt(lq2_fit_e*lq2_fit_e - lq2_meas_m*lq2_meas_m) / lq2_meas_p;
  lq2_fit_px  = scale * lq2_meas_px;
  lq2_fit_py  = scale * lq2_meas_py;
  lq2_fit_pz  = scale * lq2_meas_pz;

  // light quark 3
  lq3_fit_e = parameters[parLQ3E];
  scale = sqrt(lq3_fit_e*lq3_fit_e - lq3_meas_m*lq3_meas_m) / lq3_meas_p;
  lq3_fit_px = scale * lq3_meas_px;
  lq3_fit_py = scale * lq3_meas_py;
  lq3_fit_pz = scale * lq3_meas_pz;

  // light quark 4
  lq4_fit_e = parameters[parLQ4E];
  scale = sqrt(lq4_fit_e*lq4_fit_e - lq4_meas_m*lq4_meas_m) / lq4_meas_p;
  lq4_fit_px = scale * lq4_meas_px;
  lq4_fit_py = scale * lq4_meas_py;
  lq4_fit_pz = scale * lq4_meas_pz;

  // hadronic W 1
  whad1_fit_e  = lq1_fit_e +lq2_fit_e;
  whad1_fit_px = lq1_fit_px+lq2_fit_px;
  whad1_fit_py = lq1_fit_py+lq2_fit_py;
  whad1_fit_pz = lq1_fit_pz+lq2_fit_pz;
  whad1_fit_m = sqrt(whad1_fit_e*whad1_fit_e - (whad1_fit_px*whad1_fit_px + whad1_fit_py*whad1_fit_py + whad1_fit_pz*whad1_fit_pz));

  // hadronic W 2
  whad2_fit_e  = lq3_fit_e +lq4_fit_e;
  whad2_fit_px = lq3_fit_px+lq4_fit_px;
  whad2_fit_py = lq3_fit_py+lq4_fit_py;
  whad2_fit_pz = lq3_fit_pz+lq4_fit_pz;
  whad2_fit_m = sqrt(whad2_fit_e*whad2_fit_e - (whad2_fit_px*whad2_fit_px + whad2_fit_py*whad2_fit_py + whad2_fit_pz*whad2_fit_pz));

  // hadronic top 1
  thad1_fit_e = whad1_fit_e+bhad1_fit_e;
  thad1_fit_px = whad1_fit_px+bhad1_fit_px;
  thad1_fit_py = whad1_fit_py+bhad1_fit_py;
  thad1_fit_pz = whad1_fit_pz+bhad1_fit_pz;
  thad1_fit_m = sqrt(thad1_fit_e*thad1_fit_e - (thad1_fit_px*thad1_fit_px + thad1_fit_py*thad1_fit_py + thad1_fit_pz*thad1_fit_pz));

  // hadronic top 2
  thad2_fit_e = whad2_fit_e+bhad2_fit_e;
  thad2_fit_px = whad2_fit_px+bhad2_fit_px;
  thad2_fit_py = whad2_fit_py+bhad2_fit_py;
  thad2_fit_pz = whad2_fit_pz+bhad2_fit_pz;
  thad2_fit_m = sqrt(thad2_fit_e*thad2_fit_e - (thad2_fit_px*thad2_fit_px + thad2_fit_py*thad2_fit_py + thad2_fit_pz*thad2_fit_pz));

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTopAllHadronic::RemoveInvariantParticlePermutations() {
  // error code
  int err = 1;

  // remove the permutation from the second and the third jet
  KLFitter::Particles::ParticleType ptype = KLFitter::Particles::kParton;
  std::vector<int> indexVector_Jets;
  indexVector_Jets.push_back(2);
  indexVector_Jets.push_back(3);
  err *= (*fPermutations)->InvariantParticlePermutations(ptype, indexVector_Jets);

  indexVector_Jets.clear();
  indexVector_Jets.push_back(4);
  indexVector_Jets.push_back(5);
  err *= (*fPermutations)->InvariantParticlePermutations(ptype, indexVector_Jets);

  // remove invariant permutation when exchanging both top quarks
  std::vector<int> indexVector_JetsTop1;
  indexVector_JetsTop1.push_back(0);
  indexVector_JetsTop1.push_back(2);
  indexVector_JetsTop1.push_back(3);
  std::vector<int> indexVector_JetsTop2;
  indexVector_JetsTop2.push_back(1);
  indexVector_JetsTop2.push_back(4);
  indexVector_JetsTop2.push_back(5);
  err *= (*fPermutations)->InvariantParticleGroupPermutations(ptype, indexVector_JetsTop1, indexVector_JetsTop2);

  // remove invariant jet permutations of notevent jets
  KLFitter::Particles* particles = (*fPermutations)->Particles();
  indexVector_Jets.clear();
  for (int iPartons = 6; iPartons < particles->NPartons(); iPartons++) {
    indexVector_Jets.push_back(iPartons);
  }
  err *= (*fPermutations)->InvariantParticlePermutations(ptype, indexVector_Jets);

  // return error code
  return err;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTopAllHadronic::AdjustParameterRanges() {
  // adjust limits
  double nsigmas_jet = fFlagGetParSigmasFromTFs ? 10 : 7;

  double E = (*fParticlesPermuted)->Parton(0)->E();
  double m = fPhysicsConstants.MassBottom();
  if (fFlagUseJetMass)
    m = std::max(0.0, (*fParticlesPermuted)->Parton(0)->M());
  double sigma = fFlagGetParSigmasFromTFs ? fResEnergyBhad1->GetSigma(E) : sqrt(E);
  double Emin = std::max(m, E - nsigmas_jet* sigma);
  double Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parBhad1E, Emin, Emax);

  E = (*fParticlesPermuted)->Parton(1)->E();
  m = fPhysicsConstants.MassBottom();
  if (fFlagUseJetMass)
    m = std::max(0.0, (*fParticlesPermuted)->Parton(1)->M());
  sigma = fFlagGetParSigmasFromTFs ? fResEnergyBhad2->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parBhad2E, Emin, Emax);

  E = (*fParticlesPermuted)->Parton(2)->E();
  m = 0.001;
  if (fFlagUseJetMass)
    m = std::max(0.0, (*fParticlesPermuted)->Parton(2)->M());
  sigma = fFlagGetParSigmasFromTFs ? fResEnergyLQ1->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parLQ1E, Emin, Emax);

  E = (*fParticlesPermuted)->Parton(3)->E();
  m = 0.001;
  if (fFlagUseJetMass)
    m = std::max(0.0, (*fParticlesPermuted)->Parton(3)->M());
  sigma = fFlagGetParSigmasFromTFs ? fResEnergyLQ2->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parLQ2E, Emin, Emax);

  E = (*fParticlesPermuted)->Parton(4)->E();
  m = 0.001;
  if (fFlagUseJetMass)
    m = std::max(0.0, (*fParticlesPermuted)->Parton(4)->M());
  sigma = fFlagGetParSigmasFromTFs ? fResEnergyLQ3->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parLQ3E, Emin, Emax);

  E = (*fParticlesPermuted)->Parton(5)->E();
  m = 0.001;
  if (fFlagUseJetMass)
    m = std::max(0.0, (*fParticlesPermuted)->Parton(5)->M());
  sigma = fFlagGetParSigmasFromTFs ? fResEnergyLQ4->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parLQ4E, Emin, Emax);

  if (fFlagTopMassFixed)
    SetParameterRange(parTopM, fPhysicsConstants.MassTop(), fPhysicsConstants.MassTop());

  // no error
  return 1;
}

// ---------------------------------------------------------
void KLFitter::LikelihoodTopAllHadronic::RequestResolutionFunctions() {
  (*fDetector)->RequestResolutionType(ResolutionType::EnergyLightJet);
  (*fDetector)->RequestResolutionType(ResolutionType::EnergyBJet);
}

// ---------------------------------------------------------
double KLFitter::LikelihoodTopAllHadronic::LogLikelihood(const std::vector<double> & parameters) {
  // calculate 4-vectors
  CalculateLorentzVectors(parameters);

  // define log of likelihood
  double logprob(0.);

  // temporary flag for a safe use of the transfer functions
  bool TFgoodTmp(true);

  // jet energy resolution terms
  logprob += log(fResEnergyBhad1->p(bhad1_fit_e, bhad1_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) fTFgood = false;

  logprob += log(fResEnergyBhad2->p(bhad2_fit_e, bhad2_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) fTFgood = false;

  logprob += log(fResEnergyLQ1->p(lq1_fit_e, lq1_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) fTFgood = false;

  logprob += log(fResEnergyLQ2->p(lq2_fit_e, lq2_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) fTFgood = false;

  logprob += log(fResEnergyLQ3->p(lq3_fit_e, lq3_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) fTFgood = false;

  logprob += log(fResEnergyLQ4->p(lq4_fit_e, lq4_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) fTFgood = false;

  // physics constants
  double massW = fPhysicsConstants.MassW();
  double gammaW = fPhysicsConstants.GammaW();
  // note: top mass width should be made DEPENDENT on the top mass at a certain point
  //    fPhysicsConstants.SetMassTop(parameters[parTopM]);
  // (this will also set the correct width for the top)
  double gammaTop = fPhysicsConstants.GammaTop();

  // Breit-Wigner of hadronically decaying W-boson
  logprob += BCMath::LogBreitWignerRel(whad1_fit_m, massW, gammaW);

  // Breit-Wigner of hadronically decaying W-boson
  logprob += BCMath::LogBreitWignerRel(whad2_fit_m, massW, gammaW);

  // Breit-Wigner of first hadronically decaying top quark
  logprob += BCMath::LogBreitWignerRel(thad1_fit_m, parameters[parTopM], gammaTop);

  // Breit-Wigner of second hadronically decaying top quark
  logprob += BCMath::LogBreitWignerRel(thad2_fit_m, parameters[parTopM], gammaTop);

  // return log of likelihood
  return logprob;
}

// ---------------------------------------------------------
std::vector<double> KLFitter::LikelihoodTopAllHadronic::GetInitialParameters() {
  std::vector<double> values(GetNParameters());

  // energies of the quarks
  values[parBhad1E] = bhad1_meas_e;
  values[parBhad2E] = bhad2_meas_e;
  values[parLQ1E]  = lq1_meas_e;
  values[parLQ2E]  = lq2_meas_e;
  values[parLQ3E]  = lq3_meas_e;
  values[parLQ4E]  = lq4_meas_e;

  // still need to think about appropriate start value for top mass
  // top mass
  double mtop = (*(*fParticlesPermuted)->Parton(0) + *(*fParticlesPermuted)->Parton(2) + *(*fParticlesPermuted)->Parton(3)).M();
  if (mtop < GetParameter(parTopM)->GetLowerLimit()) {
    mtop = GetParameter(parTopM)->GetLowerLimit();
  } else if (mtop > GetParameter(parTopM)->GetUpperLimit()) {
    mtop = GetParameter(parTopM)->GetUpperLimit();
  }
  values[parTopM] = mtop;

  // return the vector
  return values;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTopAllHadronic::SavePermutedParticles() {
  bhad1_meas_e      = (*fParticlesPermuted)->Parton(0)->E();
  bhad1_meas_deteta = (*fParticlesPermuted)->DetEta(0, KLFitter::Particles::kParton);
  bhad1_meas_px     = (*fParticlesPermuted)->Parton(0)->Px();
  bhad1_meas_py     = (*fParticlesPermuted)->Parton(0)->Py();
  bhad1_meas_pz     = (*fParticlesPermuted)->Parton(0)->Pz();
  bhad1_meas_m      = SetPartonMass((*fParticlesPermuted)->Parton(0)->M(), fPhysicsConstants.MassBottom(), &bhad1_meas_px, &bhad1_meas_py, &bhad1_meas_pz, bhad1_meas_e);
  bhad1_meas_p      = sqrt(bhad1_meas_e*bhad1_meas_e - bhad1_meas_m*bhad1_meas_m);

  bhad2_meas_e      = (*fParticlesPermuted)->Parton(1)->E();
  bhad2_meas_deteta = (*fParticlesPermuted)->DetEta(1, KLFitter::Particles::kParton);
  bhad2_meas_px     = (*fParticlesPermuted)->Parton(1)->Px();
  bhad2_meas_py     = (*fParticlesPermuted)->Parton(1)->Py();
  bhad2_meas_pz     = (*fParticlesPermuted)->Parton(1)->Pz();
  bhad2_meas_m      = SetPartonMass((*fParticlesPermuted)->Parton(1)->M(), fPhysicsConstants.MassBottom(), &bhad2_meas_px, &bhad2_meas_py, &bhad2_meas_pz, bhad2_meas_e);
  bhad2_meas_p      = sqrt(bhad2_meas_e*bhad2_meas_e - bhad2_meas_m*bhad2_meas_m);

  lq1_meas_e      = (*fParticlesPermuted)->Parton(2)->E();
  lq1_meas_deteta = (*fParticlesPermuted)->DetEta(2, KLFitter::Particles::kParton);
  lq1_meas_px     = (*fParticlesPermuted)->Parton(2)->Px();
  lq1_meas_py     = (*fParticlesPermuted)->Parton(2)->Py();
  lq1_meas_pz     = (*fParticlesPermuted)->Parton(2)->Pz();
  lq1_meas_m      = SetPartonMass((*fParticlesPermuted)->Parton(2)->M(), 0., &lq1_meas_px, &lq1_meas_py, &lq1_meas_pz, lq1_meas_e);
  lq1_meas_p      = sqrt(lq1_meas_e*lq1_meas_e - lq1_meas_m*lq1_meas_m);

  lq2_meas_e      = (*fParticlesPermuted)->Parton(3)->E();
  lq2_meas_deteta = (*fParticlesPermuted)->DetEta(3, KLFitter::Particles::kParton);
  lq2_meas_px     = (*fParticlesPermuted)->Parton(3)->Px();
  lq2_meas_py     = (*fParticlesPermuted)->Parton(3)->Py();
  lq2_meas_pz     = (*fParticlesPermuted)->Parton(3)->Pz();
  lq2_meas_m      = SetPartonMass((*fParticlesPermuted)->Parton(3)->M(), 0., &lq2_meas_px, &lq2_meas_py, &lq2_meas_pz, lq2_meas_e);
  lq2_meas_p      = sqrt(lq2_meas_e*lq2_meas_e - lq2_meas_m*lq2_meas_m);

  lq3_meas_e      = (*fParticlesPermuted)->Parton(4)->E();
  lq3_meas_deteta = (*fParticlesPermuted)->DetEta(4, KLFitter::Particles::kParton);
  lq3_meas_px     = (*fParticlesPermuted)->Parton(4)->Px();
  lq3_meas_py     = (*fParticlesPermuted)->Parton(4)->Py();
  lq3_meas_pz     = (*fParticlesPermuted)->Parton(4)->Pz();
  lq3_meas_m      = SetPartonMass((*fParticlesPermuted)->Parton(4)->M(), 0., &lq3_meas_px, &lq3_meas_py, &lq3_meas_pz, lq3_meas_e);
  lq3_meas_p      = sqrt(lq3_meas_e*lq3_meas_e - lq3_meas_m*lq3_meas_m);

  lq4_meas_e      = (*fParticlesPermuted)->Parton(5)->E();
  lq4_meas_deteta = (*fParticlesPermuted)->DetEta(5, KLFitter::Particles::kParton);
  lq4_meas_px     = (*fParticlesPermuted)->Parton(5)->Px();
  lq4_meas_py     = (*fParticlesPermuted)->Parton(5)->Py();
  lq4_meas_pz     = (*fParticlesPermuted)->Parton(5)->Pz();
  lq4_meas_m      = SetPartonMass((*fParticlesPermuted)->Parton(5)->M(), 0., &lq4_meas_px, &lq4_meas_py, &lq4_meas_pz, lq4_meas_e);
  lq4_meas_p      = sqrt(lq4_meas_e*lq4_meas_e - lq4_meas_m*lq4_meas_m);

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTopAllHadronic::SaveResolutionFunctions() {
  fResEnergyBhad1 = (*fDetector)->ResEnergyBJet(bhad1_meas_deteta);
  fResEnergyBhad2 = (*fDetector)->ResEnergyBJet(bhad2_meas_deteta);
  fResEnergyLQ1  = (*fDetector)->ResEnergyLightJet(lq1_meas_deteta);
  fResEnergyLQ2  = (*fDetector)->ResEnergyLightJet(lq2_meas_deteta);
  fResEnergyLQ3  = (*fDetector)->ResEnergyLightJet(lq3_meas_deteta);
  fResEnergyLQ4  = (*fDetector)->ResEnergyLightJet(lq4_meas_deteta);

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTopAllHadronic::BuildModelParticles() {
  if (GetBestFitParameters().size() > 0) CalculateLorentzVectors(GetBestFitParameters());

  TLorentzVector * bhad1 = fParticlesModel->Parton(0);
  TLorentzVector * bhad2 = fParticlesModel->Parton(1);
  TLorentzVector * lq1  = fParticlesModel->Parton(2);
  TLorentzVector * lq2  = fParticlesModel->Parton(3);
  TLorentzVector * lq3  = fParticlesModel->Parton(4);
  TLorentzVector * lq4  = fParticlesModel->Parton(5);

  TLorentzVector * whad1  = fParticlesModel->Boson(0);
  TLorentzVector * whad2  = fParticlesModel->Boson(1);
  TLorentzVector * thad1  = fParticlesModel->Parton(6);
  TLorentzVector * thad2  = fParticlesModel->Parton(7);

  bhad1->SetPxPyPzE(bhad1_fit_px, bhad1_fit_py, bhad1_fit_pz, bhad1_fit_e);
  bhad2->SetPxPyPzE(bhad2_fit_px, bhad2_fit_py, bhad2_fit_pz, bhad2_fit_e);
  lq1 ->SetPxPyPzE(lq1_fit_px,  lq1_fit_py,  lq1_fit_pz,  lq1_fit_e);
  lq2 ->SetPxPyPzE(lq2_fit_px,  lq2_fit_py,  lq2_fit_pz,  lq2_fit_e);
  lq3 ->SetPxPyPzE(lq3_fit_px,  lq3_fit_py,  lq3_fit_pz,  lq3_fit_e);
  lq4 ->SetPxPyPzE(lq4_fit_px,  lq4_fit_py,  lq4_fit_pz,  lq4_fit_e);

  (*whad1) = (*lq1)  + (*lq2);
  (*whad2) = (*lq3)  + (*lq4);
  (*thad1) = (*whad1) + (*bhad1);
  (*thad2) = (*whad2) + (*bhad2);

  // no error
  return 1;
}

// ---------------------------------------------------------
std::vector<double> KLFitter::LikelihoodTopAllHadronic::LogLikelihoodComponents(std::vector<double> parameters) {
  std::vector<double> vecci;

  // calculate 4-vectors
  CalculateLorentzVectors(parameters);

  // temporary flag for a safe use of the transfer functions
  bool TFgoodTmp(true);

  // jet energy resolution terms
  vecci.push_back(log(fResEnergyBhad1->p(bhad1_fit_e, bhad1_meas_e, &TFgoodTmp)));  // comp0
  if (!TFgoodTmp) fTFgood = false;

  vecci.push_back(log(fResEnergyBhad2->p(bhad2_fit_e, bhad2_meas_e, &TFgoodTmp)));  // comp1
  if (!TFgoodTmp) fTFgood = false;

  vecci.push_back(log(fResEnergyLQ1->p(lq1_fit_e, lq1_meas_e, &TFgoodTmp)));  // comp2
  if (!TFgoodTmp) fTFgood = false;

  vecci.push_back(log(fResEnergyLQ2->p(lq2_fit_e, lq2_meas_e, &TFgoodTmp)));  // comp3
  if (!TFgoodTmp) fTFgood = false;

  vecci.push_back(log(fResEnergyLQ3->p(lq3_fit_e, lq3_meas_e, &TFgoodTmp)));  // comp4
  if (!TFgoodTmp) fTFgood = false;

  vecci.push_back(log(fResEnergyLQ4->p(lq4_fit_e, lq4_meas_e, &TFgoodTmp)));  // comp5
  if (!TFgoodTmp) fTFgood = false;

  // physics constants
  double massW = fPhysicsConstants.MassW();
  double gammaW = fPhysicsConstants.GammaW();
  // note: top mass width should be made DEPENDENT on the top mass at a certain point
  //    fPhysicsConstants.SetMassTop(parameters[parTopM]);
  // (this will also set the correct width for the top)
  double gammaTop = fPhysicsConstants.GammaTop();

  // Breit-Wigner of hadronically decaying W-boson1
  vecci.push_back(BCMath::LogBreitWignerRel(whad1_fit_m, massW, gammaW));  // comp6

  // Breit-Wigner of hadronically decaying W-boson2
  vecci.push_back(BCMath::LogBreitWignerRel(whad2_fit_m, massW, gammaW));  // comp7

  // Breit-Wigner of hadronically decaying top quark1
  vecci.push_back(BCMath::LogBreitWignerRel(thad1_fit_m, parameters[parTopM], gammaTop));  // comp8

  // Breit-Wigner of hadronically decaying top quark
  vecci.push_back(BCMath::LogBreitWignerRel(thad2_fit_m, parameters[parTopM], gammaTop));  // comp9

  // return log of likelihood
  return vecci;
}
