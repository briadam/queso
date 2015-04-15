#include <queso/Environment.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/UniformVectorRV.h>
#include <queso/StatisticalInverseProblem.h>
#include <queso/ScalarFunction.h>
#include <queso/VectorSet.h>

template<class V, class M>
class Likelihood : public QUESO::BaseScalarFunction<V, M>
{
public:

  Likelihood(const char * prefix, const QUESO::VectorSet<V, M> & domain)
    : QUESO::BaseScalarFunction<V, M>(prefix, domain)
  {
  }

  virtual ~Likelihood()
  {
  }

  virtual double lnValue(const V & domainVector, const V * domainDirection,
      V * gradVector, M * hessianMatrix, V * hessianEffect) const
  {
    // undo the scaling transformation
    double x1 = 2.0*domainVector[0];
    double x2 = 2.0*domainVector[1];
    double resid1 = 10.0*(x2-x1*x1);
    double resid2 = 1.0-x1;
    double misfit = resid1*resid1 + resid2*resid2;
    return -0.5 * misfit;
  }

  virtual double actualValue(const V & domainVector, const V * domainDirection,
      V * gradVector, M * hessianMatrix, V * hessianEffect) const
  {
    return std::exp(this->lnValue(domainVector, domainDirection, gradVector,
          hessianMatrix, hessianEffect));
  }
};

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);

  QUESO::FullEnvironment env(MPI_COMM_WORLD, argv[1], "", NULL);

  // 2 vars
  QUESO::VectorSpace<QUESO::GslVector, QUESO::GslMatrix> paramSpace(env,
      "param_", 2, NULL);

  QUESO::GslVector paramMins(paramSpace.zeroVector());
  QUESO::GslVector paramMaxs(paramSpace.zeroVector());

  // Original domain [-2, 2] --> [-1, 1]
  double min_val = -2.0/2.0;
  double max_val = 2.0/2.0;
  paramMins.cwSet(min_val);
  paramMaxs.cwSet(max_val);

  QUESO::BoxSubset<QUESO::GslVector, QUESO::GslMatrix> paramDomain("param_",
      paramSpace, paramMins, paramMaxs);

  QUESO::UniformVectorRV<QUESO::GslVector, QUESO::GslMatrix> priorRv("prior_",
      paramDomain);

  Likelihood<QUESO::GslVector, QUESO::GslMatrix> lhood("llhd_", paramDomain);

  QUESO::GenericVectorRV<QUESO::GslVector, QUESO::GslMatrix>
    postRv("post_", paramSpace);

  QUESO::StatisticalInverseProblem<QUESO::GslVector, QUESO::GslMatrix>
    ip("", NULL, priorRv, lhood, postRv);

  QUESO::GslVector paramInitials(paramSpace.zeroVector());
  // Scaled initial value
  paramInitials[0] = -0.5; 
  paramInitials[1] = 0.5;  

  QUESO::GslMatrix proposalCovMatrix(paramSpace.zeroVector());

  proposalCovMatrix(0, 0) = 0.125;
  proposalCovMatrix(0, 1) = -0.25;
  proposalCovMatrix(1, 0) = -0.25;
  proposalCovMatrix(1, 1) = 0.50125;

  QUESO::MhOptionsValues* calIpMhOptionsValues = new QUESO::MhOptionsValues();

  // Dakota setttings aren't needed to reproduce the error

  // calIpMhOptionsValues->m_rawChainSize = 1000;
  // calIpMhOptionsValues->m_putOutOfBoundsInChain       = false;

  // calIpMhOptionsValues->m_drMaxNumExtraStages = 1;
  // calIpMhOptionsValues->m_drScalesForExtraStages.resize(1);
  // calIpMhOptionsValues->m_drScalesForExtraStages[0] = 5;

  // calIpMhOptionsValues->m_amInitialNonAdaptInterval = 1;
  // calIpMhOptionsValues->m_amAdaptInterval           = 100;
  // calIpMhOptionsValues->m_amEta                     = 2.88;
  // calIpMhOptionsValues->m_amEpsilon                 = 1.e-8;

  // Occurs with or without LogitTransform

  // calIpMhOptionsValues->m_doLogitTransform = true;

  ip.solveWithBayesMetropolisHastings(calIpMhOptionsValues, paramInitials,
				      &proposalCovMatrix);

  MPI_Finalize();

  return 0;
}
