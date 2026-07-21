#include "sditerator.h"

#define raddeg 57.2957795130823229
#define ln10 2.30258509299404590
#define secIn1200m 4.0027691424e-6 // Seconds in a [1200m] unit

#define DS1_TO_DS2_DATE 81111

//     FOR ANALYZING TA SD DATA IN C++
//************************************************************************
///////////////////////////////////////////////////////////////////////


//static double pdErr(double theta, double dtheta, double dphi)
//{
//  return
//    sqrt(dtheta*dtheta+sin(theta/raddeg)*sin(theta/raddeg)*dphi*dphi);
//}

/////////// EXECUTED ON EACH EVENT //////////////////////
//void cppanalysis(FILE *outFl)
void cppanalysis()
{
  int i;
  int x;
  
  
  /* Only print triggered events */
  if (rusdraw_.nofwf>0){
    printf("#EVENT DATA\n");
    printf("%d %e %f %f %f %f %f %d %d %d %d %d ",
	   rusdmc_.parttype, rusdmc_.energy, rusdmc_.theta, rusdmc_.phi, rusdmc_.corexyz[0], rusdmc_.corexyz[1], rusdmc_.corexyz[2],
	   rusdraw_.yymmdd, rusdraw_.hhmmss, rufptn_.nstclust, rusdraw_.nofwf, rusdraw_.usec);
    printf("%f %f %f %f %d %f %f %f %f %f ",
	   rufldf_.energy[0], rufldf_.sc[0], rufldf_.dsc[0],
	   rufldf_.chi2[0], rufldf_.ndof[0], 
	   //rusdgeom_.chi2[1], rusdgeom_.ndof[1], 
	   rufldf_.xcore[0], rufldf_.dxcore[0], rufldf_.ycore[0], rufldf_.dycore[0], rufldf_.s800[0]); // relates LDF fit
    printf("%f %f %f %f %f %d %f %f ",
	   rusdgeom_.theta[1], rusdgeom_.phi[1], rusdgeom_.dtheta[1], rusdgeom_.dphi[1], rusdgeom_.chi2[1],
	   rusdgeom_.ndof[1], rusdgeom_.t0[1], rusdgeom_.dt0[1]); // relates geometry fit
    printf("%f %f ",
	   rufldf_.bdist, rufldf_.tdist); // for border cut
    printf("%f %f %f %f %f %d %f %f %f %f\n",
	   rusdgeom_.theta[2], rusdgeom_.phi[2], rusdgeom_.dtheta[2], rusdgeom_.dphi[2],
	   rusdgeom_.chi2[2], rusdgeom_.ndof[2], rusdgeom_.t0[2], rusdgeom_.dt0[2],
	   rusdgeom_.a, rusdgeom_.da); // free-curvature geometry fit + curvature

    //printf("#SD DATA\n");
    printf("#SD meta DATA\n");
    for(x=0; x<rufptn_.nhits; x++)
      {
	printf("%04d %02d %f %f %f %f %f %f %f %f %f %d ",
	       rufptn_.xxyy[x], rufptn_.isgood[x],
	       rufptn_.reltime[x][0], rufptn_.reltime[x][1],
	       rufptn_.pulsa[x][0], rufptn_.pulsa[x][1],
	       rufptn_.xyzclf[x][0], rufptn_.xyzclf[x][1], rufptn_.xyzclf[x][2],
	       rufptn_.vem[x][0], rufptn_.vem[x][1], rufptn_.nfold[x]);
      }
    printf("\n");
    printf("#SD waveform DATA\n");
    for(x=0; x<rusdraw_.nofwf; x++)
      {
	printf("%04d %08d %08d ",
	       rusdraw_.xxyy[x], rusdraw_.clkcnt[x], rusdraw_.mclkcnt[x]);
	for(i=0;i<128;i++)
          {
            printf("%d ",rusdraw_.fadc[x][0][i]); //lower FADC counts
          }
	for(i=0;i<128;i++)
          {
            printf("%d ",rusdraw_.fadc[x][1][i]); //upper FADC counts
          }
      }
    printf("\n");
    printf("#badsdinfo\n");
    for(x=0; x<bsdinfo_.nsdsout; x++)
      {
	printf("%04d %d ",bsdinfo_.xxyyout[x],bsdinfo_.bitfout[x]);
      }
    printf("\n");
  }
  //printf("== END EVENT ==\n");
  /*
  int passed_cuts;

  double gfchi2pdof;
  double ldfchi2pdof;
 
  passed_cuts = 1;

  // number of good SDs cut
  if (rufptn_.nstclust < 4)
    passed_cuts = 0;

  // Distance from the surrounding edgre should be less than 1200m
  // ( 1 counter separation unit ) 
  if ( rufldf_.bdist < 1.0 )
    passed_cuts = 0;

  // T-Shape boundary CUT
  if ( (rusdraw_.yymmdd < DS1_TO_DS2_DATE) && (rufldf_.tdist < 1.0) )
    passed_cuts = 0;
  
  // Geometry fit chi2 / dof cut
  if (rusdgeom_.ndof[1] > 0)
    {
      gfchi2pdof = rusdgeom_.chi2[1] / (double) rusdgeom_.ndof[1];
    }
  else
    {
      gfchi2pdof = rusdgeom_.chi2[1];
    }
  if (gfchi2pdof > 4.0)
    passed_cuts = 0;
  
  // LDF Fit chi2 / dof
  if (rufldf_.ndof[1] > 0)
    {
      ldfchi2pdof = rufldf_.chi2[0] / (double) rufldf_.ndof[0];
    }
  else
    {
      ldfchi2pdof = rufldf_.chi2[0];
    }
  if (ldfchi2pdof > 4.0)
    passed_cuts = 0;
  

  // Pointing direction error cut ( should be less than 5 degrees ) 
  if (pdErr(rusdgeom_.theta[1], rusdgeom_.dtheta[1], rusdgeom_.dphi[1]) > 5.0)
    passed_cuts = 0;
  
  // Cut on resolution of LDF scaling constant.  SAME as 
  // as cut on resolution of S800. 
  if (rufldf_.dsc[0] / rufldf_.sc[0] > 0.25)
    passed_cuts = 0;
  
  // Zenith angle cut:
  if (rusdgeom_.theta[1] > 45.0 )
    passed_cuts = 0;
  
  
  // Energy cut ? 


  if ( passed_cuts ) 
    {
      fprintf (outFl, "PASSED CUTS: ");
    }
  else
    {
      fprintf (outFl, "DID NOT PASS CUTS: ");
    }
  
  fprintf (
	   outFl, "%06d %06d.%06d %.2f %.2f %.2f\n",
	   rusdraw_.yymmdd,rusdraw_.hhmmss,rusdraw_.usec,
	   rusdgeom_.theta[1],rusdgeom_.phi[1],rufldf_.energy[0]
	   );
  */  
 
}
