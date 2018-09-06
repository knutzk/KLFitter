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

#include "KLFitter/LikelihoodTopLeptonJetsUDSep.h"

#include <iostream>
#include <algorithm>

#include "BAT/BCMath.h"
#include "BAT/BCParameter.h"
#include "KLFitter/DetectorBase.h"
#include "KLFitter/ParticleCollection.h"
#include "KLFitter/Permutations.h"
#include "KLFitter/PhysicsConstants.h"
#include "KLFitter/ResolutionBase.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLorentzVector.h"

namespace KLFitter {
// ---------------------------------------------------------
LikelihoodTopLeptonJetsUDSep::LikelihoodTopLeptonJetsUDSep()
  : LikelihoodTopLeptonJets::LikelihoodTopLeptonJets()
  , m_ljet_separation_method(LikelihoodTopLeptonJetsUDSep::kNone) {
  // define model particles
  this->DefineModelParticles();

  // define parameters
  this->DefineParameters();
  }

// ---------------------------------------------------------
LikelihoodTopLeptonJetsUDSep::~LikelihoodTopLeptonJetsUDSep() = default;

// ---------------------------------------------------------
int LikelihoodTopLeptonJetsUDSep::DefineModelParticles() {
  // create the particles of the model
  m_particles_model.reset(new ParticleCollection{});

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
  parton2.SetTrueFlavor(Particles::PartonTrueFlavor::kLightUp);
  m_particles_model->AddParticle(parton2);

  Particles::Parton parton3{"light quark 2", TLorentzVector{}};
  parton3.SetIdentifier(3);
  parton3.SetTrueFlavor(Particles::PartonTrueFlavor::kLightDown);
  m_particles_model->AddParticle(parton3);

  if (m_lepton_type == kElectron) {
    m_particles_model->AddParticle(Particles::Electron{"electron", TLorentzVector{}});
  } else if (m_lepton_type == kMuon) {
    m_particles_model->AddParticle(Particles::Muon{"muon", TLorentzVector{}});
  }

  m_particles_model->AddParticle(Particles::Neutrino{"neutrino", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Boson{"hadronic W", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Boson{"leptonic W", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Parton{"hadronic top", TLorentzVector{}});
  m_particles_model->AddParticle(Particles::Parton{"leptonic top", TLorentzVector{}});

  // no error
  return 1;
}

// ---------------------------------------------------------
void LikelihoodTopLeptonJetsUDSep::DefineParameters() {
  // rename light quark parameters
  this->GetParameter("energy light quark 1")->SetName("energy light up type quark");
  this->GetParameter("energy light quark 2")->SetName("energy light down type quark");
}

// ---------------------------------------------------------
int LikelihoodTopLeptonJetsUDSep::RemoveInvariantParticlePermutations() {
  // error code
  int err = 1;

  Particles::Type ptype = Particles::Type::kParton;
  std::vector<int> indexVector_Jets;
  // remove invariant jet permutations of all jets not considered
  const ParticleCollection* particles = (*m_permutations)->Particles();
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
double LikelihoodTopLeptonJetsUDSep::LogEventProbability() {
  double logprob = 0;

  if (fBTagMethod != kNotag) {
    double logprobbtag = LogEventProbabilityBTag();
    if (logprobbtag <= -1e99) return -1e99;
    logprob += logprobbtag;
  }
  if (m_ljet_separation_method != kNone) {
    double logprobljetweight = LogEventProbabilityLJetReweight();
    if (logprobljetweight <= -1e99) return -1e99;
    logprob += logprobljetweight;
  }

  // use integrated value of LogLikelihood (default)
  if (fFlagIntegrate) {
    logprob += log(GetIntegral());
  } else {
    logprob += LogLikelihood(GetBestFitParameters());
  }

  return logprob;
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::LogEventProbabilityLJetReweight() {
  double logprob = 0;
  switch (m_ljet_separation_method) {
  case kPermReweight:

    if (!(m_up_jet_pt_histo && m_down_jet_pt_histo&& m_bjet_pt_histo && m_up_jet_tag_weight_histo && m_down_jet_tag_weight_histo && m_bjet_tag_weight_histo)) {
      std::cout <<  " KLFitter::LikelihoodTopLeptonJetsUDSep::LogEventProbabilityLJetReweight() : Histograms were not set properly! " << std::endl;
      return -1e99;
    }

    for (size_t i = 0; i < m_particles_model->partons.size(); ++i) {
      // get index of corresponding measured particle.

      int index = m_particles_model->partons.at(i).GetIdentifier();

      if (index < 0) {
        continue;
      }
      if (!((*m_particles_permuted)->partons.at(index).GetBTagWeightIsSet())) {
        std::cout <<  " KLFitter::LikelihoodTopLeptonJetsUDSep::LogEventProbabilityLJetReweight() : bTag weight for particle was not set ! " << std::endl;
        return -1e99;
      }
      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(i).GetTrueFlavor();
      if (trueFlavor == Particles::PartonTrueFlavor::kB) {
        logprob += log(BJetPt((*m_particles_permuted)->GetP4(Particles::Type::kParton, index)->Pt()));
        logprob += log(BJetTagWeight((*m_particles_permuted)->partons.at(index).GetBTagWeight()));
      }
      if (trueFlavor == Particles::PartonTrueFlavor::kLightUp) {
        logprob += log(UpJetPt((*m_particles_permuted)->GetP4(Particles::Type::kParton, index)->Pt()));
        logprob += log(UpJetTagWeight((*m_particles_permuted)->partons.at(index).GetBTagWeight()));
      }
      if (trueFlavor == Particles::PartonTrueFlavor::kLightDown) {
        logprob += log(DownJetPt((*m_particles_permuted)->GetP4(Particles::Type::kParton, index)->Pt()));
        logprob += log(DownJetTagWeight((*m_particles_permuted)->partons.at(index).GetBTagWeight()));
      }
    }
    return logprob;
    break;

  case kPermReweight2D:
    if (!(m_up_jet_2d_weight_histo && m_down_jet_2d_weight_histo && m_bjet_2d_weight_histo)) {
      std::cout <<  " KLFitter::LikelihoodTopLeptonJetsUDSep::LogEventProbabilityLJetReweight() : 2D Histograms were not set properly! " << std::endl;
      return -1e99;
    }

    for (size_t i = 0; i < m_particles_model->partons.size(); ++i) {
      // get index of corresponding measured particle.

      int index = m_particles_model->partons.at(i).GetIdentifier();

      if (index < 0) {
        continue;
      }
      if (!((*m_particles_permuted)->partons.at(index).GetBTagWeightIsSet())) {
        std::cout <<  " KLFitter::LikelihoodTopLeptonJetsUDSep::LogEventProbabilityLJetReweight() : bTag weight for particle was not set ! " << std::endl;
        return -1e99;
      }
      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(i).GetTrueFlavor();
      if (trueFlavor == Particles::PartonTrueFlavor::kB) {
        logprob += log(BJetProb((*m_particles_permuted)->partons.at(index).GetBTagWeight(), (*m_particles_permuted)->GetP4(Particles::Type::kParton, index)->Pt()));
      }
      if (trueFlavor == Particles::PartonTrueFlavor::kLightUp) {
        logprob += log(UpJetProb((*m_particles_permuted)->partons.at(index).GetBTagWeight(), (*m_particles_permuted)->GetP4(Particles::Type::kParton, index)->Pt()));
      }
      if (trueFlavor == Particles::PartonTrueFlavor::kLightDown) {
        logprob += log(DownJetProb((*m_particles_permuted)->partons.at(index).GetBTagWeight(), (*m_particles_permuted)->GetP4(Particles::Type::kParton, index)->Pt()));
      }
    }
    return logprob;
    break;

  default:
    return logprob;
    break;
  }
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::LogEventProbabilityBTag() {
  double logprob = 0;

  double probbtag = 1;

  if (fBTagMethod == kVeto) {
    // loop over all model particles.  calculate the overall b-tagging
    // probability which is the product of all probabilities.
    for (size_t i = 0; i < m_particles_model->partons.size(); ++i) {
      // get index of corresponding measured particle.
      int index = m_particles_model->partons.at(i).GetIdentifier();
      if (index < 0)
        continue;

      Particles::PartonTrueFlavor trueFlavor = m_particles_model->partons.at(i).GetTrueFlavor();
      bool isBTagged = m_particles_model->partons.at(i).GetIsBTagged();
      if (((trueFlavor == Particles::PartonTrueFlavor::kLightUp) || (trueFlavor == Particles::PartonTrueFlavor::kLightDown)) && isBTagged == true)
        probbtag = 0.;
    }

    if (probbtag > 0) {
      logprob += log(probbtag);
    } else {
      return -1e99;
    }
  } else if (fBTagMethod == kWorkingPoint) {
    // loop over all model particles.  calculate the overall b-tagging
    // probability which is the product of all probabilities.
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
        std::cout <<  " KLFitter::LikelihoodTopLeptonJetsUDSep::LogEventProbability() : Your working points are not set properly! Returning 0 probability " << std::endl;
        return -1e99;
      }

      if (((trueFlavor == Particles::PartonTrueFlavor::kLightUp) || (trueFlavor == Particles::PartonTrueFlavor::kLightDown)) && isBTagged) {
        logprob += log(1./rejection);
      } else if (((trueFlavor == Particles::PartonTrueFlavor::kLightUp) || (trueFlavor == Particles::PartonTrueFlavor::kLightDown)) && !isBTagged) {
        logprob += log(1 - 1./rejection);
      } else if (trueFlavor == Particles::PartonTrueFlavor::kB && isBTagged) {
        logprob += log(efficiency);
      } else if (trueFlavor == Particles::PartonTrueFlavor::kB && !isBTagged) {
        logprob += log(1 - efficiency);
      } else {
        std::cout << " KLFitter::LikelihoodTopLeptonJetsUDSep::LogEventProbability() : b-tagging association failed! " << std::endl;
      }
    }
  }

  return logprob;
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::UpJetPt(double pt) {
  return m_up_jet_pt_histo->GetBinContent(m_up_jet_pt_histo->GetXaxis()->FindBin(pt));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::DownJetPt(double pt) {
  return m_down_jet_pt_histo->GetBinContent(m_down_jet_pt_histo->GetXaxis()->FindBin(pt));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::BJetPt(double pt) {
  return m_bjet_pt_histo->GetBinContent(m_bjet_pt_histo->GetXaxis()->FindBin(pt));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::UpJetTagWeight(double tagweight) {
  return m_up_jet_tag_weight_histo->GetBinContent(m_up_jet_tag_weight_histo->GetXaxis()->FindBin(tagweight));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::DownJetTagWeight(double tagweight) {
  return m_down_jet_tag_weight_histo->GetBinContent(m_down_jet_tag_weight_histo->GetXaxis()->FindBin(tagweight));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::BJetTagWeight(double tagweight) {
  return m_bjet_tag_weight_histo->GetBinContent(m_bjet_tag_weight_histo->GetXaxis()->FindBin(tagweight));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::UpJetProb(double tagweight, double pt) {
  return m_up_jet_2d_weight_histo->GetBinContent(m_up_jet_2d_weight_histo->GetXaxis()->FindBin(tagweight), m_up_jet_2d_weight_histo->GetYaxis()->FindBin(pt));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::DownJetProb(double tagweight, double pt) {
  return m_down_jet_2d_weight_histo->GetBinContent(m_down_jet_2d_weight_histo->GetXaxis()->FindBin(tagweight), m_down_jet_2d_weight_histo->GetYaxis()->FindBin(pt));
}

// ---------------------------------------------------------
double LikelihoodTopLeptonJetsUDSep::BJetProb(double tagweight, double pt) {
  return m_bjet_2d_weight_histo->GetBinContent(m_bjet_2d_weight_histo->GetXaxis()->FindBin(tagweight), m_bjet_2d_weight_histo->GetYaxis()->FindBin(pt));
}

// ---------------------------------------------------------
int LikelihoodTopLeptonJetsUDSep::LHInvariantPermutationPartner(int iperm, int nperms, int *switchpar1, int *switchpar2) {
  int partnerid = -1;
  int cache = iperm % 6;
  switch (nperms) {
  case 24:
    if (iperm % 2) {
      partnerid = iperm - 1;
    } else {
      partnerid = iperm + 1;
    }
    break;

  case 120:
    if (cache > 2) {
      partnerid = iperm - 3;
    } else {
      partnerid = iperm + 3;
    }
    break;

  default: partnerid = -1;
  }
  *switchpar1 = 2;
  *switchpar2 = 3;
  return partnerid;
}
}  // namespace KLFitter
