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

#include "KLFitter/LikelihoodTTZTrilepton.h"

#include <algorithm>
#include <iostream>

#include "BAT/BCMath.h"
#include "BAT/BCParameter.h"
#include "KLFitter/DetectorBase.h"
#include "KLFitter/ParticleCollection.h"
#include "KLFitter/Permutations.h"
#include "KLFitter/PhysicsConstants.h"
#include "KLFitter/ResolutionBase.h"
#include "TLorentzVector.h"

// ---------------------------------------------------------
KLFitter::LikelihoodTTZTrilepton::LikelihoodTTZTrilepton()
  : KLFitter::LikelihoodBase::LikelihoodBase()
  , m_flag_top_mass_fixed(false)
  , m_flag_get_par_sigmas_from_TFs(false)
  , m_et_miss_x(0.)
  , m_et_miss_y(0.)
  , m_et_miss_sum(0.)
  , m_lepton_type(kElectron)
  , fInvMassCutoff(5.)
  , fOnShellFraction(0.869) {
  // define model particles
  this->DefineModelParticles();

  // define parameters
  this->DefineParameters();
}

// ---------------------------------------------------------
KLFitter::LikelihoodTTZTrilepton::~LikelihoodTTZTrilepton() = default;

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::SetET_miss_XY_SumET(double etx, double ety, double sumet) {
  // set missing ET x and y component and the m_et_miss_sum
  m_et_miss_x = etx;
  m_et_miss_y = ety;
  m_et_miss_sum = sumet;

  // no error
  return 1;
}

// ---------------------------------------------------------
void KLFitter::LikelihoodTTZTrilepton::RequestResolutionFunctions() {
  (*m_detector)->RequestResolutionType(ResolutionType::EnergyLightJet);
  (*m_detector)->RequestResolutionType(ResolutionType::EnergyBJet);
  (*m_detector)->RequestResolutionType(ResolutionType::EnergyElectron);
  (*m_detector)->RequestResolutionType(ResolutionType::EnergyMuon);
  (*m_detector)->RequestResolutionType(ResolutionType::MissingET);
}

// ---------------------------------------------------------
void KLFitter::LikelihoodTTZTrilepton::SetLeptonType(LeptonType leptontype) {
  if (leptontype != kElectron && leptontype != kMuon) {
    std::cout << "KLFitter::SetLeptonTyp(). Warning: lepton type not defined. Set electron as lepton type." << std::endl;
    m_lepton_type = kElectron;
  } else {
    m_lepton_type = leptontype;
  }

  // define model particles
  DefineModelParticles();
}

// ---------------------------------------------------------
void KLFitter::LikelihoodTTZTrilepton::SetLeptonType(int leptontype) {
  if (leptontype != 1 && leptontype != 2) {
    std::cout << "KLFitter::SetLeptonTyp(). Warning: lepton type not defined. Set electron as lepton type." << std::endl;
    leptontype = 1;
  }

  if (leptontype == 1) {
    SetLeptonType(kElectron);
  } else if (leptontype == 2) {
    SetLeptonType(kMuon);
  }
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::DefineModelParticles() {
  // create the particles of the model
  m_particles_model.reset(new KLFitter::ParticleCollection{});

  // add model particles
  Particles::Parton parton0{"hadronic b quark", TLorentzVector{}};
  parton0.SetIdentifier(0);
  parton0.SetTrueFlavor(Particles::PartonTrueFlavor::kB);
  m_particles_model->AddParticle(parton0);

  Particles::Parton parton1{"leptonic b quark", TLorentzVector{}};
  parton1.SetIdentifier(1);
  parton1.SetTrueFlavor(Particles::PartonTrueFlavor::kB);
  m_particles_model->AddParticle(parton1);

  Particles::Parton parton2{"light quark 1", TLorentzVector{}};
  parton2.SetIdentifier(2);
  parton2.SetTrueFlavor(Particles::PartonTrueFlavor::kLight);
  m_particles_model->AddParticle(parton2);

  Particles::Parton parton3{"light quark 2", TLorentzVector{}};
  parton3.SetIdentifier(3);
  parton3.SetTrueFlavor(Particles::PartonTrueFlavor::kLight);
  m_particles_model->AddParticle(parton3);

  if (m_lepton_type == kElectron) {
    m_particles_model->AddParticle(Particles::Electron{"electron", TLorentzVector{}});
    m_particles_model->AddParticle(Particles::Electron{"electron Z1", TLorentzVector{}});
    m_particles_model->AddParticle(Particles::Electron{"electron Z2", TLorentzVector{}});
  } else if (m_lepton_type == kMuon) {
    m_particles_model->AddParticle(Particles::Muon{"muon", TLorentzVector{}});
    m_particles_model->AddParticle(Particles::Muon{"muon Z1", TLorentzVector{}});
    m_particles_model->AddParticle(Particles::Muon{"muon Z2", TLorentzVector{}});
  }

  m_particles_model->AddParticle(Particles::Neutrino{"neutrino", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Boson{"hadronic W", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Boson{"leptonic W", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Parton{"hadronic top", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Parton{"leptonic top", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Boson{"Z boson", TLorentzVector{}});

  // no error
  return 1;
}

// ---------------------------------------------------------
void KLFitter::LikelihoodTTZTrilepton::DefineParameters() {
  // add parameters of model
  AddParameter("energy hadronic b",       m_physics_constants.MassBottom(), 1000.0);  // parBhadE
  AddParameter("energy leptonic b",       m_physics_constants.MassBottom(), 1000.0);  // parBlepE
  AddParameter("energy light quark 1",    0.0, 1000.0);                              // parLQ1E
  AddParameter("energy light quark 2",    0.0, 1000.0);                              // parLQ2E
  AddParameter("energy lepton",           0.0, 1000.0);                              // parLepE
  AddParameter("energy Z lepton 1",       0.0, 1000.0);                              // parLepZ1E
  AddParameter("energy Z lepton 2",       0.0, 1000.0);                              // parLepZ2E
  AddParameter("p_x neutrino",        -1000.0, 1000.0);                              // parNuPx
  AddParameter("p_y neutrino",        -1000.0, 1000.0);                              // parNuPy
  AddParameter("p_z neutrino",        -1000.0, 1000.0);                              // parNuPz
  AddParameter("top mass",              100.0, 1000.0);                              // parTopM
  AddParameter("Z mass",                  0.0, 1000.0);                              // parZM
}

// ---------------------------------------------------------
double KLFitter::LikelihoodTTZTrilepton::LogBreitWignerRelNorm(const double& x, const double& mean, const double& gamma) {
  double g = std::sqrt(pow(mean, 2) * (pow(mean, 2) + pow(gamma, 2)));
  double k = (2 * std::sqrt(2) * mean * gamma * g) / (M_PI * std::sqrt(pow(mean, 2) + g));
  double f = k / (pow(pow(x, 2) - pow(mean, 2), 2) + pow(mean * gamma, 2));
  return log(f);
}

// ---------------------------------------------------------
double KLFitter::LikelihoodTTZTrilepton::LogZCombinedDistribution(const double& x, const double& mean, const double& gamma) {
  // note: This catches exceptions when the variables are set to non-sensible
  // values. If there is any common way to handle exceptions, it should be
  // implemented here.
  const auto fraction = fOnShellFraction;
  if (fraction < 0 || fraction > 1) throw;
  if (fInvMassCutoff < 0) throw;

  double g = std::sqrt(pow(mean, 2) * (pow(mean, 2) + pow(gamma, 2)));
  double k = (2 * std::sqrt(2) * mean * gamma * g) / (M_PI * std::sqrt(pow(mean, 2) + g));
  double on_shell = k / (pow(pow(x, 2) - pow(mean, 2), 2) + pow(mean * gamma, 2));
  double off_shell = fInvMassCutoff / x / x;
  return log(on_shell * fraction + off_shell * (1 - fraction));
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::CalculateLorentzVectors(std::vector <double> const& parameters) {
  static double scale;
  static double whad_fit_e;
  static double whad_fit_px;
  static double whad_fit_py;
  static double whad_fit_pz;
  static double wlep_fit_e;
  static double wlep_fit_px;
  static double wlep_fit_py;
  static double wlep_fit_pz;
  static double thad_fit_e;
  static double thad_fit_px;
  static double thad_fit_py;
  static double thad_fit_pz;
  static double tlep_fit_e;
  static double tlep_fit_px;
  static double tlep_fit_py;
  static double tlep_fit_pz;

  static double Z_fit_e;
  static double Z_fit_px;
  static double Z_fit_py;
  static double Z_fit_pz;

  // hadronic b quark
  bhad_fit_e = parameters[parBhadE];
  scale = sqrt(bhad_fit_e*bhad_fit_e - bhad_meas_m*bhad_meas_m) / bhad_meas_p;
  bhad_fit_px = scale * bhad_meas_px;
  bhad_fit_py = scale * bhad_meas_py;
  bhad_fit_pz = scale * bhad_meas_pz;

  // leptonic b quark
  blep_fit_e = parameters[parBlepE];
  scale = sqrt(blep_fit_e*blep_fit_e - blep_meas_m*blep_meas_m) / blep_meas_p;
  blep_fit_px = scale * blep_meas_px;
  blep_fit_py = scale * blep_meas_py;
  blep_fit_pz = scale * blep_meas_pz;

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

  // Z lepton 1
  lepZ1_fit_e = parameters[parLepZ1E];
  scale = lepZ1_fit_e / lepZ1_meas_e;
  lepZ1_fit_px = scale * lepZ1_meas_px;
  lepZ1_fit_py = scale * lepZ1_meas_py;
  lepZ1_fit_pz = scale * lepZ1_meas_pz;

  // Z lepton 2
  lepZ2_fit_e = parameters[parLepZ2E];
  scale = lepZ2_fit_e / lepZ2_meas_e;
  lepZ2_fit_px = scale * lepZ2_meas_px;
  lepZ2_fit_py = scale * lepZ2_meas_py;
  lepZ2_fit_pz = scale * lepZ2_meas_pz;

  // lepton
  lep_fit_e = parameters[parLepE];
  scale = lep_fit_e / lep_meas_e;
  lep_fit_px = scale * lep_meas_px;
  lep_fit_py = scale * lep_meas_py;
  lep_fit_pz = scale * lep_meas_pz;

  // neutrino
  nu_fit_px = parameters[parNuPx];
  nu_fit_py = parameters[parNuPy];
  nu_fit_pz = parameters[parNuPz];
  nu_fit_e  = sqrt(nu_fit_px*nu_fit_px + nu_fit_py*nu_fit_py + nu_fit_pz*nu_fit_pz);

  // hadronic W
  whad_fit_e  = lq1_fit_e +lq2_fit_e;
  whad_fit_px = lq1_fit_px+lq2_fit_px;
  whad_fit_py = lq1_fit_py+lq2_fit_py;
  whad_fit_pz = lq1_fit_pz+lq2_fit_pz;
  whad_fit_m = sqrt(whad_fit_e*whad_fit_e - (whad_fit_px*whad_fit_px + whad_fit_py*whad_fit_py + whad_fit_pz*whad_fit_pz));

  // leptonic W
  wlep_fit_e  = lep_fit_e +nu_fit_e;
  wlep_fit_px = lep_fit_px+nu_fit_px;
  wlep_fit_py = lep_fit_py+nu_fit_py;
  wlep_fit_pz = lep_fit_pz+nu_fit_pz;
  wlep_fit_m = sqrt(wlep_fit_e*wlep_fit_e - (wlep_fit_px*wlep_fit_px + wlep_fit_py*wlep_fit_py + wlep_fit_pz*wlep_fit_pz));

  // hadronic top
  thad_fit_e = whad_fit_e+bhad_fit_e;
  thad_fit_px = whad_fit_px+bhad_fit_px;
  thad_fit_py = whad_fit_py+bhad_fit_py;
  thad_fit_pz = whad_fit_pz+bhad_fit_pz;
  thad_fit_m = sqrt(thad_fit_e*thad_fit_e - (thad_fit_px*thad_fit_px + thad_fit_py*thad_fit_py + thad_fit_pz*thad_fit_pz));

  // leptonic top
  tlep_fit_e = wlep_fit_e+blep_fit_e;
  tlep_fit_px = wlep_fit_px+blep_fit_px;
  tlep_fit_py = wlep_fit_py+blep_fit_py;
  tlep_fit_pz = wlep_fit_pz+blep_fit_pz;
  tlep_fit_m = sqrt(tlep_fit_e*tlep_fit_e - (tlep_fit_px*tlep_fit_px + tlep_fit_py*tlep_fit_py + tlep_fit_pz*tlep_fit_pz));

  // Z boson
  Z_fit_e  = lepZ1_fit_e  + lepZ2_fit_e;
  Z_fit_px = lepZ1_fit_px + lepZ2_fit_px;
  Z_fit_py = lepZ1_fit_py + lepZ2_fit_py;
  Z_fit_pz = lepZ1_fit_pz + lepZ2_fit_pz;
  Z_fit_m  = sqrt(Z_fit_e*Z_fit_e - (Z_fit_px*Z_fit_px + Z_fit_py*Z_fit_py + Z_fit_pz*Z_fit_pz));

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::RemoveInvariantParticlePermutations() {
  // error code
  int err = 1;

  // remove the permutation from the second and the third jet
  Particles::Type ptype = Particles::Type::kParton;
  std::vector<int> indexVector_Jets;
  indexVector_Jets.push_back(2);
  indexVector_Jets.push_back(3);
  err *= (*m_permutations)->InvariantParticlePermutations(ptype, indexVector_Jets);

  // remove the permutation from the two Z leptons
  Particles::Type ptypeLepZ;
  if (m_lepton_type == kElectron) {
    ptypeLepZ = Particles::Type::kElectron;
  } else {
    ptypeLepZ = Particles::Type::kMuon;
  }

  std::vector<int> indexVector_LepZ;
  indexVector_LepZ.push_back(1);
  indexVector_LepZ.push_back(2);
  err *= (*m_permutations)->InvariantParticlePermutations(ptypeLepZ, indexVector_LepZ);

  // remove invariant jet permutations of notevent jets
  const KLFitter::ParticleCollection* particles = (*m_permutations)->Particles();
  indexVector_Jets.clear();
  for (size_t iPartons = 4; iPartons < particles->partons.size(); iPartons++) {
    indexVector_Jets.push_back(iPartons);
  }
  err *= (*m_permutations)->InvariantParticlePermutations(ptype, indexVector_Jets);

  // remove the permutation from the other lepton
  if (m_lepton_type == kElectron) {
    ptype = Particles::Type::kMuon;
    std::vector<int> indexVector_Muons;
    for (size_t iMuon = 0; iMuon < particles->muons.size(); iMuon++) {
      indexVector_Muons.push_back(iMuon);
    }
    err *= (*m_permutations)->InvariantParticlePermutations(ptype, indexVector_Muons);
  } else if (m_lepton_type == kMuon) {
    ptype = Particles::Type::kElectron;
    std::vector<int> indexVector_Electrons;
    for (size_t iElectron = 0; iElectron < particles->electrons.size(); iElectron++) {
      indexVector_Electrons.push_back(iElectron);
    }
    err *= (*m_permutations)->InvariantParticlePermutations(ptype, indexVector_Electrons);
  }

  // return error code
  return err;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::AdjustParameterRanges() {
  // adjust limits
  double nsigmas_jet    = m_flag_get_par_sigmas_from_TFs ? 10 : 7;
  double nsigmas_lepton = m_flag_get_par_sigmas_from_TFs ? 10 : 2;
  double nsigmas_met    = m_flag_get_par_sigmas_from_TFs ? 10 : 1;

  double E = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 0)->E();
  double m = m_physics_constants.MassBottom();
  if (m_use_jet_mass)
    m = std::max(0.0, (*m_particles_permuted)->GetP4(Particles::Type::kParton, 0)->M());
  double sigma = m_flag_get_par_sigmas_from_TFs ? fResEnergyBhad->GetSigma(E) : sqrt(E);
  double Emin = std::max(m, E - nsigmas_jet* sigma);
  double Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parBhadE, Emin, Emax);

  E = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 1)->E();
  m = m_physics_constants.MassBottom();
  if (m_use_jet_mass)
    m = std::max(0.0, (*m_particles_permuted)->GetP4(Particles::Type::kParton, 1)->M());
  sigma = m_flag_get_par_sigmas_from_TFs ? fResEnergyBlep->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parBlepE, Emin, Emax);

  E = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 2)->E();
  m = 0.001;
  if (m_use_jet_mass)
    m = std::max(0.0, (*m_particles_permuted)->GetP4(Particles::Type::kParton, 2)->M());
  sigma = m_flag_get_par_sigmas_from_TFs ? fResEnergyLQ1->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parLQ1E, Emin, Emax);

  E = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)->E();
  m = 0.001;
  if (m_use_jet_mass)
    m = std::max(0.0, (*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)->M());
  sigma = m_flag_get_par_sigmas_from_TFs ? fResEnergyLQ2->GetSigma(E) : sqrt(E);
  Emin = std::max(m, E - nsigmas_jet* sigma);
  Emax  = E + nsigmas_jet* sigma;
  SetParameterRange(parLQ2E, Emin, Emax);

  if (m_lepton_type == kElectron) {
    E = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 0)->E();
    sigma = m_flag_get_par_sigmas_from_TFs ? fResLepton->GetSigma(E) : sqrt(E);
    Emin = std::max(0.001, E - nsigmas_lepton* sigma);
    Emax  = E + nsigmas_lepton* sigma;
  } else if (m_lepton_type == kMuon) {
    E = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0)->E();
    double sintheta = sin((*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0)->Theta());
    sigma = m_flag_get_par_sigmas_from_TFs ? fResLepton->GetSigma(E*sintheta)/sintheta : E*E*sintheta;
    double sigrange = nsigmas_lepton* sigma;
    Emin = std::max(0.001, E -sigrange);
    Emax = E +sigrange;
  }
  SetParameterRange(parLepE, Emin, Emax);

  if (m_lepton_type == kElectron) {
    E = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 1)->E();
    sigma = m_flag_get_par_sigmas_from_TFs ? fResLeptonZ1->GetSigma(E) : sqrt(E);
    Emin = std::max(0.001, E - nsigmas_lepton* sigma);
    Emax  = E + nsigmas_lepton* sigma;
  } else if (m_lepton_type == kMuon) {
    E = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 1)->E();
    double sintheta = sin((*m_particles_permuted)->GetP4(Particles::Type::kMuon, 1)->Theta());
    sigma = m_flag_get_par_sigmas_from_TFs ? fResLeptonZ1->GetSigma(E*sintheta)/sintheta : E*E*sintheta;
    double sigrange = nsigmas_lepton* sigma;
    Emin = std::max(0.001, E -sigrange);
    Emax = E +sigrange;
  }
  SetParameterRange(parLepZ1E, Emin, Emax);

  if (m_lepton_type == kElectron) {
    E = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 2)->E();
    sigma = m_flag_get_par_sigmas_from_TFs ? fResLeptonZ2->GetSigma(E) : sqrt(E);
    Emin = std::max(0.001, E - nsigmas_lepton* sigma);
    Emax  = E + nsigmas_lepton* sigma;
  } else if (m_lepton_type == kMuon) {
    E = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 2)->E();
    double sintheta = sin((*m_particles_permuted)->GetP4(Particles::Type::kMuon, 2)->Theta());
    sigma = m_flag_get_par_sigmas_from_TFs ? fResLeptonZ2->GetSigma(E*sintheta)/sintheta : E*E*sintheta;
    double sigrange = nsigmas_lepton* sigma;
    Emin = std::max(0.001, E -sigrange);
    Emax = E +sigrange;
  }
  SetParameterRange(parLepZ2E, Emin, Emax);

  // note: this is hard-coded at the moment

  sigma = m_flag_get_par_sigmas_from_TFs ? fResMET->GetSigma(m_et_miss_sum) : 100;
  double sigrange = nsigmas_met*sigma;
  SetParameterRange(parNuPx, m_et_miss_x-sigrange, m_et_miss_x+sigrange);
  SetParameterRange(parNuPy, m_et_miss_y-sigrange, m_et_miss_y+sigrange);

  if (m_flag_top_mass_fixed)
    SetParameterRange(parTopM, m_physics_constants.MassTop(), m_physics_constants.MassTop());

  SetParameterRange(parZM, m_physics_constants.MassZ(), m_physics_constants.MassZ());

  // no error
  return 1;
}

// ---------------------------------------------------------
double KLFitter::LikelihoodTTZTrilepton::LogLikelihood(const std::vector<double> & parameters) {
  // calculate 4-vectors
  CalculateLorentzVectors(parameters);

  // define log of likelihood
  double logprob(0.);

  // temporary flag for a safe use of the transfer functions
  bool TFgoodTmp(true);

  // jet energy resolution terms
  logprob += log(fResEnergyBhad->p(bhad_fit_e, bhad_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) m_TFs_are_good = false;

  logprob += log(fResEnergyBlep->p(blep_fit_e, blep_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) m_TFs_are_good = false;

  logprob += log(fResEnergyLQ1->p(lq1_fit_e, lq1_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) m_TFs_are_good = false;

  logprob += log(fResEnergyLQ2->p(lq2_fit_e, lq2_meas_e, &TFgoodTmp));
  if (!TFgoodTmp) m_TFs_are_good = false;

  // lepton energy resolution terms
  if (m_lepton_type == kElectron) {
    logprob += log(fResLepton->p(lep_fit_e, lep_meas_e, &TFgoodTmp));
  } else if (m_lepton_type == kMuon) {
    logprob += log(fResLepton->p(lep_fit_e* lep_meas_sintheta, lep_meas_pt, &TFgoodTmp));
  }
  if (!TFgoodTmp) m_TFs_are_good = false;

  if (m_lepton_type == kElectron) {
    logprob += log(fResLeptonZ1->p(lepZ1_fit_e, lepZ1_meas_e, &TFgoodTmp));
  } else if (m_lepton_type == kMuon) {
    logprob += log(fResLeptonZ1->p(lepZ1_fit_e* lepZ1_meas_sintheta, lepZ1_meas_pt, &TFgoodTmp));
  }
  if (!TFgoodTmp) m_TFs_are_good = false;

  if (m_lepton_type == kElectron) {
    logprob += log(fResLeptonZ2->p(lepZ2_fit_e, lepZ2_meas_e, &TFgoodTmp));
  } else if (m_lepton_type == kMuon) {
    logprob += log(fResLeptonZ2->p(lepZ2_fit_e* lepZ2_meas_sintheta, lepZ2_meas_pt, &TFgoodTmp));
  }
  if (!TFgoodTmp) m_TFs_are_good = false;

  // neutrino px and py
  logprob += log(fResMET->p(nu_fit_px, m_et_miss_x, &TFgoodTmp, m_et_miss_sum));
  if (!TFgoodTmp) m_TFs_are_good = false;

  logprob += log(fResMET->p(nu_fit_py, m_et_miss_y, &TFgoodTmp, m_et_miss_sum));
  if (!TFgoodTmp) m_TFs_are_good = false;

  // physics constants
  double massW = m_physics_constants.MassW();
  double gammaW = m_physics_constants.GammaW();
  // note: top mass width should be made DEPENDENT on the top mass at a certain point
  //    m_physics_constants.SetMassTop(parameters[parTopM]);
  // (this will also set the correct width for the top)
  double gammaTop = m_physics_constants.GammaTop();

  double gammaZ = m_physics_constants.GammaZ();

  // note: as opposed to the LikelihoodTopLeptonJets class, we use a normalised
  // version of the Breit-Wigner here to make sure that weightings between
  // functions are handled correctly.

  // Breit-Wigner of hadronically decaying W-boson
  logprob += LogBreitWignerRelNorm(whad_fit_m, massW, gammaW);

  // Breit-Wigner of leptonically decaying W-boson
  logprob += LogBreitWignerRelNorm(wlep_fit_m, massW, gammaW);

  // Breit-Wigner of hadronically decaying top quark
  logprob += LogBreitWignerRelNorm(thad_fit_m, parameters[parTopM], gammaTop);

  // Breit-Wigner of leptonically decaying top quark
  logprob += LogBreitWignerRelNorm(tlep_fit_m, parameters[parTopM], gammaTop);

  // Breit-Wigner of Z boson decaying into two leptons
  logprob += LogZCombinedDistribution(Z_fit_m, parameters[parZM], gammaZ);

  // return log of likelihood
  return logprob;
}

// ---------------------------------------------------------
std::vector<double> KLFitter::LikelihoodTTZTrilepton::GetInitialParameters() {
  std::vector<double> values(GetInitialParametersWoNeutrinoPz());

  // check second neutrino solution
  std::vector<double> neutrino_pz_solutions = GetNeutrinoPzSolutions();
  if (neutrino_pz_solutions.size() == 1) {
    values[parNuPz] = neutrino_pz_solutions[0];
  } else if (neutrino_pz_solutions.size() == 2) {
    double sol1, sol2;
    values[parNuPz] = neutrino_pz_solutions[0];
    sol1 = LogLikelihood(values);
    values[parNuPz] = neutrino_pz_solutions[1];
    sol2 = LogLikelihood(values);

    if (sol1 > sol2)
      values[parNuPz] = neutrino_pz_solutions[0];
  }

  return values;
}

// ---------------------------------------------------------
std::vector<double> KLFitter::LikelihoodTTZTrilepton::GetInitialParametersWoNeutrinoPz() {
  std::vector<double> values(GetNParameters());

  // energies of the quarks
  values[parBhadE] = bhad_meas_e;
  values[parBlepE] = blep_meas_e;
  values[parLQ1E]  = lq1_meas_e;
  values[parLQ2E]  = lq2_meas_e;
  values[parLepZ1E]  = lepZ1_meas_e;
  values[parLepZ2E]  = lepZ2_meas_e;

  // energy of the lepton
  if (m_lepton_type == kElectron) {
    values[parLepE] = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 0)->E();
  } else if (m_lepton_type == kMuon) {
    values[parLepE] = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0)->E();
  }

  if (m_lepton_type == kElectron) {
    values[parLepZ1E] = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 1)->E();
  } else if (m_lepton_type == kMuon) {
    values[parLepZ1E] = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 1)->E();
  }

  if (m_lepton_type == kElectron) {
    values[parLepZ2E] = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 2)->E();
  } else if (m_lepton_type == kMuon) {
    values[parLepZ2E] = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 2)->E();
  }

  // missing px and py
  values[parNuPx] = m_et_miss_x;
  values[parNuPy] = m_et_miss_y;

  // pz of the neutrino
  values[parNuPz] = 0.;

  // top mass
  double mtop = (*(*m_particles_permuted)->GetP4(Particles::Type::kParton, 0) + *(*m_particles_permuted)->GetP4(Particles::Type::kParton, 2) + *(*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)).M();
  if (mtop < GetParameter(parTopM)->GetLowerLimit()) {
    mtop = GetParameter(parTopM)->GetLowerLimit();
  } else if (mtop > GetParameter(parTopM)->GetUpperLimit()) {
    mtop = GetParameter(parTopM)->GetUpperLimit();
  }
  values[parTopM] = mtop;

  // Z mass
  double mz;
  if (m_lepton_type == kElectron) {
    mz = (*(*m_particles_permuted)->GetP4(Particles::Type::kElectron, 1) + *(*m_particles_permuted)->GetP4(Particles::Type::kElectron, 2)).M();
  } else {
    mz = (*(*m_particles_permuted)->GetP4(Particles::Type::kMuon, 1) + *(*m_particles_permuted)->GetP4(Particles::Type::kMuon, 2)).M();
  }
  if (mz < GetParameter(parZM)->GetLowerLimit()) {
    mz = GetParameter(parZM)->GetLowerLimit();
  } else if (mz > GetParameter(parZM)->GetUpperLimit()) {
    mz = GetParameter(parZM)->GetUpperLimit();
  }
  values[parZM] = mz;

  // return the vector
  return values;
}

// ---------------------------------------------------------
std::vector<double> KLFitter::LikelihoodTTZTrilepton::GetNeutrinoPzSolutions() {
  return CalculateNeutrinoPzSolutions();
}

// ---------------------------------------------------------
std::vector<double> KLFitter::LikelihoodTTZTrilepton::CalculateNeutrinoPzSolutions(TLorentzVector* additionalParticle) {
  std::vector<double> pz;

  KLFitter::PhysicsConstants constants;
  // electron mass
  double mE = 0.;

  double px_c = 0.0;
  double py_c = 0.0;
  double pz_c = 0.0;
  double Ec = 0.0;

  if (m_lepton_type == kElectron) {
    px_c = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 0)->Px();
    py_c = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 0)->Py();
    pz_c = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 0)->Pz();
    Ec = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 0)->E();
  } else if (m_lepton_type == kMuon) {
    px_c = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0)->Px();
    py_c = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0)->Py();
    pz_c = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0)->Pz();
    Ec = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0)->E();
  }

  // add additional particle to "charged lepton" 4-vector
  if (additionalParticle) {
    px_c += additionalParticle->Px();
    py_c += additionalParticle->Py();
    pz_c += additionalParticle->Pz();
    Ec += additionalParticle->E();
  }

  double px_nu = m_et_miss_x;
  double py_nu = m_et_miss_y;
  double alpha = constants.MassW()*constants.MassW() - mE*mE + 2*(px_c*px_nu + py_c*py_nu);

  double a = pz_c*pz_c - Ec*Ec;
  double b = alpha* pz_c;
  double c = - Ec*Ec* (px_nu*px_nu + py_nu*py_nu) + alpha*alpha/4.;

  double discriminant = b*b - 4*a*c;
  if (discriminant < 0.)
    return pz;

  double pz_offset = - b / (2*a);

  double squareRoot = sqrt(discriminant);
  if (squareRoot < 1.e-6) {
    pz.push_back(pz_offset);
  } else {
    pz.push_back(pz_offset + squareRoot / (2*a));
    pz.push_back(pz_offset - squareRoot / (2*a));
  }

  return pz;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::SavePermutedParticles() {
  bhad_meas_e      = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 0)->E();
  bhad_meas_deteta = (*m_particles_permuted)->partons.at(0).GetDetEta();
  bhad_meas_px     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 0)->Px();
  bhad_meas_py     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 0)->Py();
  bhad_meas_pz     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 0)->Pz();
  bhad_meas_m      = SetPartonMass((*m_particles_permuted)->GetP4(Particles::Type::kParton, 0)->M(), m_physics_constants.MassBottom(), &bhad_meas_px, &bhad_meas_py, &bhad_meas_pz, bhad_meas_e);
  bhad_meas_p      = sqrt(bhad_meas_e*bhad_meas_e - bhad_meas_m*bhad_meas_m);

  blep_meas_e      = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 1)->E();
  blep_meas_deteta = (*m_particles_permuted)->partons.at(1).GetDetEta();
  blep_meas_px     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 1)->Px();
  blep_meas_py     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 1)->Py();
  blep_meas_pz     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 1)->Pz();
  blep_meas_m      = SetPartonMass((*m_particles_permuted)->GetP4(Particles::Type::kParton, 1)->M(), m_physics_constants.MassBottom(), &blep_meas_px, &blep_meas_py, &blep_meas_pz, blep_meas_e);
  blep_meas_p      = sqrt(blep_meas_e*blep_meas_e - blep_meas_m*blep_meas_m);

  lq1_meas_e      = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 2)->E();
  lq1_meas_deteta = (*m_particles_permuted)->partons.at(2).GetDetEta();
  lq1_meas_px     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 2)->Px();
  lq1_meas_py     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 2)->Py();
  lq1_meas_pz     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 2)->Pz();
  lq1_meas_m      = SetPartonMass((*m_particles_permuted)->GetP4(Particles::Type::kParton, 2)->M(), 0., &lq1_meas_px, &lq1_meas_py, &lq1_meas_pz, lq1_meas_e);
  lq1_meas_p      = sqrt(lq1_meas_e*lq1_meas_e - lq1_meas_m*lq1_meas_m);

  lq2_meas_e      = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)->E();
  lq2_meas_deteta = (*m_particles_permuted)->partons.at(3).GetDetEta();
  lq2_meas_px     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)->Px();
  lq2_meas_py     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)->Py();
  lq2_meas_pz     = (*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)->Pz();
  lq2_meas_m      = SetPartonMass((*m_particles_permuted)->GetP4(Particles::Type::kParton, 3)->M(), 0., &lq2_meas_px, &lq2_meas_py, &lq2_meas_pz, lq2_meas_e);
  lq2_meas_p      = sqrt(lq2_meas_e*lq2_meas_e - lq2_meas_m*lq2_meas_m);

  TLorentzVector * leptonZ1(0);
  TLorentzVector * leptonZ2(0);
  if (m_lepton_type == kElectron) {
    leptonZ1 = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 1);
    lepZ1_meas_deteta = (*m_particles_permuted)->electrons.at(1).GetDetEta();
    leptonZ2 = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 2);
    lepZ2_meas_deteta = (*m_particles_permuted)->electrons.at(2).GetDetEta();
  } else {
    leptonZ1 = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 1);
    lepZ1_meas_deteta = (*m_particles_permuted)->muons.at(1).GetDetEta();
    leptonZ2 = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 2);
    lepZ2_meas_deteta = (*m_particles_permuted)->muons.at(2).GetDetEta();
  }

  lepZ1_meas_e        = leptonZ1->E();
  lepZ1_meas_sintheta = sin(leptonZ1->Theta());
  lepZ1_meas_pt       = leptonZ1->Pt();
  lepZ1_meas_px       = leptonZ1->Px();
  lepZ1_meas_py       = leptonZ1->Py();
  lepZ1_meas_pz       = leptonZ1->Pz();

  lepZ2_meas_e        = leptonZ2->E();
  lepZ2_meas_sintheta = sin(leptonZ2->Theta());
  lepZ2_meas_pt       = leptonZ2->Pt();
  lepZ2_meas_px       = leptonZ2->Px();
  lepZ2_meas_py       = leptonZ2->Py();
  lepZ2_meas_pz       = leptonZ2->Pz();

  TLorentzVector * lepton(0);
  if (m_lepton_type == kElectron) {
    lepton = (*m_particles_permuted)->GetP4(Particles::Type::kElectron, 0);
    lep_meas_deteta = (*m_particles_permuted)->electrons.at(0).GetDetEta();
  } else {
    lepton = (*m_particles_permuted)->GetP4(Particles::Type::kMuon, 0);
    lep_meas_deteta = (*m_particles_permuted)->muons.at(0).GetDetEta();
  }
  lep_meas_e        = lepton->E();
  lep_meas_sintheta = sin(lepton->Theta());
  lep_meas_pt       = lepton->Pt();
  lep_meas_px       = lepton->Px();
  lep_meas_py       = lepton->Py();
  lep_meas_pz       = lepton->Pz();

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::SaveResolutionFunctions() {
  fResEnergyBhad = (*m_detector)->ResEnergyBJet(bhad_meas_deteta);
  fResEnergyBlep = (*m_detector)->ResEnergyBJet(blep_meas_deteta);
  fResEnergyLQ1  = (*m_detector)->ResEnergyLightJet(lq1_meas_deteta);
  fResEnergyLQ2  = (*m_detector)->ResEnergyLightJet(lq2_meas_deteta);
  if (m_lepton_type == kElectron) {
    fResLepton = (*m_detector)->ResEnergyElectron(lep_meas_deteta);
  } else if (m_lepton_type == kMuon) {
    fResLepton = (*m_detector)->ResEnergyMuon(lep_meas_deteta);
  }
  fResMET = (*m_detector)->ResMissingET();

  if (m_lepton_type == kElectron) {
    fResLeptonZ1 = (*m_detector)->ResEnergyElectron(lepZ1_meas_deteta);
  } else if (m_lepton_type == kMuon) {
    fResLeptonZ1 = (*m_detector)->ResEnergyMuon(lepZ1_meas_deteta);
  }

  if (m_lepton_type == kElectron) {
    fResLeptonZ2 = (*m_detector)->ResEnergyElectron(lepZ2_meas_deteta);
  } else if (m_lepton_type == kMuon) {
    fResLeptonZ2 = (*m_detector)->ResEnergyMuon(lepZ2_meas_deteta);
  }
  fResMET = (*m_detector)->ResMissingET();

  // no error
  return 1;
}

// ---------------------------------------------------------
int KLFitter::LikelihoodTTZTrilepton::BuildModelParticles() {
  if (GetBestFitParameters().size() > 0) CalculateLorentzVectors(GetBestFitParameters());

  TLorentzVector * bhad = m_particles_model->GetP4(Particles::Type::kParton, 0);
  TLorentzVector * blep = m_particles_model->GetP4(Particles::Type::kParton, 1);
  TLorentzVector * lq1  = m_particles_model->GetP4(Particles::Type::kParton, 2);
  TLorentzVector * lq2  = m_particles_model->GetP4(Particles::Type::kParton, 3);
  TLorentzVector * lep(0);
  if (m_lepton_type == kElectron) {
    lep  = m_particles_model->GetP4(Particles::Type::kElectron, 0);
  } else if (m_lepton_type == kMuon) {
    lep  = m_particles_model->GetP4(Particles::Type::kMuon, 0);
  }
  TLorentzVector * lepZ1(0);
  if (m_lepton_type == kElectron) {
    lepZ1  = m_particles_model->GetP4(Particles::Type::kElectron, 1);
  } else if (m_lepton_type == kMuon) {
    lepZ1  = m_particles_model->GetP4(Particles::Type::kMuon, 1);
  }
  TLorentzVector * lepZ2(0);
  if (m_lepton_type == kElectron) {
    lepZ2  = m_particles_model->GetP4(Particles::Type::kElectron, 2);
  } else if (m_lepton_type == kMuon) {
    lepZ2  = m_particles_model->GetP4(Particles::Type::kMuon, 2);
  }
  TLorentzVector * nu   = m_particles_model->GetP4(Particles::Type::kNeutrino, 0);
  TLorentzVector * whad  = m_particles_model->GetP4(Particles::Type::kBoson, 0);
  TLorentzVector * wlep  = m_particles_model->GetP4(Particles::Type::kBoson, 1);
  TLorentzVector * thad  = m_particles_model->GetP4(Particles::Type::kParton, 4);
  TLorentzVector * tlep  = m_particles_model->GetP4(Particles::Type::kParton, 5);

  TLorentzVector * Z = m_particles_model->GetP4(Particles::Type::kBoson, 2);

  bhad->SetPxPyPzE(bhad_fit_px, bhad_fit_py, bhad_fit_pz, bhad_fit_e);
  blep->SetPxPyPzE(blep_fit_px, blep_fit_py, blep_fit_pz, blep_fit_e);
  lq1 ->SetPxPyPzE(lq1_fit_px,  lq1_fit_py,  lq1_fit_pz,  lq1_fit_e);
  lq2 ->SetPxPyPzE(lq2_fit_px,  lq2_fit_py,  lq2_fit_pz,  lq2_fit_e);
  lepZ1->SetPxPyPzE(lepZ1_fit_px, lepZ1_fit_py, lepZ1_fit_pz, lepZ1_fit_e);
  lepZ2->SetPxPyPzE(lepZ2_fit_px, lepZ2_fit_py, lepZ2_fit_pz, lepZ2_fit_e);
  lep ->SetPxPyPzE(lep_fit_px,  lep_fit_py,  lep_fit_pz,  lep_fit_e);
  nu  ->SetPxPyPzE(nu_fit_px,   nu_fit_py,   nu_fit_pz,   nu_fit_e);

  (*whad) = (*lq1)  + (*lq2);
  (*wlep) = (*lep)  + (*nu);
  (*thad) = (*whad) + (*bhad);
  (*tlep) = (*wlep) + (*blep);

  (*Z) = (*lepZ1) + (*lepZ2);

  // no error
  return 1;
}

// ---------------------------------------------------------
std::vector<double> KLFitter::LikelihoodTTZTrilepton::LogLikelihoodComponents(std::vector<double> parameters) {
  std::vector<double> vecci;

  // calculate 4-vectors
  CalculateLorentzVectors(parameters);

  // temporary flag for a safe use of the transfer functions
  bool TFgoodTmp(true);

  // jet energy resolution terms
  vecci.push_back(log(fResEnergyBhad->p(bhad_fit_e, bhad_meas_e, &TFgoodTmp)));  // comp0
  if (!TFgoodTmp) m_TFs_are_good = false;

  vecci.push_back(log(fResEnergyBlep->p(blep_fit_e, blep_meas_e, &TFgoodTmp)));  // comp1
  if (!TFgoodTmp) m_TFs_are_good = false;

  vecci.push_back(log(fResEnergyLQ1->p(lq1_fit_e, lq1_meas_e, &TFgoodTmp)));  // comp2
  if (!TFgoodTmp) m_TFs_are_good = false;

  vecci.push_back(log(fResEnergyLQ2->p(lq2_fit_e, lq2_meas_e, &TFgoodTmp)));  // comp3
  if (!TFgoodTmp) m_TFs_are_good = false;

  // lepton energy resolution terms
  if (m_lepton_type == kElectron) {
    vecci.push_back(log(fResLepton->p(lep_fit_e, lep_meas_e, &TFgoodTmp)));  // comp4
  } else if (m_lepton_type == kMuon) {
    vecci.push_back(log(fResLepton->p(lep_fit_e* lep_meas_sintheta, lep_meas_pt, &TFgoodTmp)));  // comp4
  }
  if (!TFgoodTmp) m_TFs_are_good = false;

  if (m_lepton_type == kElectron) {
    vecci.push_back(log(fResLeptonZ1->p(lepZ1_fit_e, lepZ1_meas_e, &TFgoodTmp)));  // comp4
  } else if (m_lepton_type == kMuon) {
    vecci.push_back(log(fResLeptonZ1->p(lepZ1_fit_e* lepZ1_meas_sintheta, lepZ1_meas_pt, &TFgoodTmp)));  // comp4
  }
  if (!TFgoodTmp) m_TFs_are_good = false;

  if (m_lepton_type == kElectron) {
    vecci.push_back(log(fResLeptonZ2->p(lepZ2_fit_e, lepZ2_meas_e, &TFgoodTmp)));  // comp4
  } else if (m_lepton_type == kMuon) {
    vecci.push_back(log(fResLeptonZ2->p(lepZ2_fit_e* lepZ2_meas_sintheta, lepZ2_meas_pt, &TFgoodTmp)));  // comp4
  }
  if (!TFgoodTmp) m_TFs_are_good = false;

  // neutrino px and py
  vecci.push_back(log(fResMET->p(nu_fit_px, m_et_miss_x, &TFgoodTmp, m_et_miss_sum)));  // comp5
  if (!TFgoodTmp) m_TFs_are_good = false;

  vecci.push_back(log(fResMET->p(nu_fit_py, m_et_miss_y, &TFgoodTmp, m_et_miss_sum)));  // comp6
  if (!TFgoodTmp) m_TFs_are_good = false;

  // physics constants
  double massW = m_physics_constants.MassW();
  double gammaW = m_physics_constants.GammaW();
  // note: top mass width should be made DEPENDENT on the top mass at a certain point
  //    m_physics_constants.SetMassTop(parameters[parTopM]);
  // (this will also set the correct width for the top)
  double gammaTop = m_physics_constants.GammaTop();

  double gammaZ = m_physics_constants.GammaZ();

  // note: as opposed to the LikelihoodTopLeptonJets class, we use a normalised
  // version of the Breit-Wigner here to make sure that weightings between
  // functions are handled correctly.

  // Breit-Wigner of hadronically decaying W-boson
  vecci.push_back(LogBreitWignerRelNorm(whad_fit_m, massW, gammaW));  // comp7

  // Breit-Wigner of leptonically decaying W-boson
  vecci.push_back(LogBreitWignerRelNorm(wlep_fit_m, massW, gammaW));  // comp8

  // Breit-Wigner of hadronically decaying top quark
  vecci.push_back(LogBreitWignerRelNorm(thad_fit_m, parameters[parTopM], gammaTop));  // comp9

  // Breit-Wigner of leptonically decaying top quark
  vecci.push_back(LogBreitWignerRelNorm(tlep_fit_m, parameters[parTopM], gammaTop));  // comp10

  // Breit-Wigner of Z decaying into 2 leptons
  vecci.push_back(LogZCombinedDistribution(Z_fit_m, parameters[parZM], gammaZ));  // comp11

  // return log of likelihood
  return vecci;
}
