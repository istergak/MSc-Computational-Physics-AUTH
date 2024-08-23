#include <cmath>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <time.h>
#include "TRint.h"
#include "TCanvas.h"
#include "TROOT.h"    // For gROOT
#include "TSystem.h"  // For gSystem
#include "Rtypes.h"   // For Double_t, Int_t
#include "TH1D.h"     // For TH1D
#include "TLatex.h"   // For TLatex
#include "TEllipse.h" // For TEllipse
#include "TMarker.h"  // For TMarker
#include "TLine.h"    // For TLine
#include "TGClient.h" // For ROOT Client
#include "TApplication.h"

using namespace std;

#define NR_END 1

// MSc Computational Physics AUTH
// Compuational Particle Physics
// Academic Year: 2023-2024
// Semester 2
// Implemented by: Ioannis Stergakis
// AEM: 4439

// ATLAS Muon Detector Simulation for ΔΕΘ 2024
// Version 3.2 (.cpp script to be compiled to .exe)
// Version characteristics:
// 1) NO intro or outro page
// 2) N0 Satistics for the drift time of the trajectories
// 3) Printing the simulation progress on the ROOT Session prompt
// 4) Includes the TApplication class for the compilation of the script to .exe app

// Windows compilation instructions:
// 1) Open the x64 Native Tools Command prompt from the start menu
// 2) Go to the directory of the current script using the 'cd' command
// 3) Copy paste the following command (without the //):

// cl -nologo -MD -GR /EHsc /std:c++17 -Zc:__cplusplus muon_atlas_v3_2.cpp -I %ROOTSYS%\include /link -LIBPATH:%ROOTSYS%\lib 
// libCore.lib libGpad.lib libHist.lib libGui.lib libRIO.lib libGraf.lib libGraf3d.lib libMathCore.lib libTree.lib libRint.lib 
// libPostscript.lib libMatrix.lib

// Notes: Before the compilation make sure you have downloaded a ROOT version for Windows and at least Visual Studio Community to 
// gain to access the native tools command prompt. Create an environment variable named ROOTSYS and store at it the path where the ROOT 
// app is stored (something like C:\root_v6.32.02). Also include the root directory to the System's path.

// Random number generator function prototype
double randf(double,double);

// Arbitrary size vector creation function prototype
double **dmatrix(int,int);

// Free memory of the 2D dmatrix prototype function
int free_dmatrix(double **,int);

// Showing intro function prototype
void showintro(TCanvas *c, Double_t, Double_t, Double_t,Double_t,Double_t,Double_t);

// Showing outro function prototype
void showoutro(TCanvas *c, Double_t,Double_t, Double_t, Double_t,Double_t,Double_t,Int_t);

// Tubes drawing function prototype
double **tubes(double,int,int,int,double,double, double,double **);

// Muon trajectory and detection tubes drawing function prototype
std::vector <double> muon_traj(TCanvas *c,double,double,double, double,double,double,double **,double);




// Main function body
int main(int argc, char **argv){
// Defining the app
TApplication *app = new TApplication("MyApp", &argc, argv);

// Reseting ROOT   
gROOT->Reset();

// Initializing random number generator
time_t t;
srand((unsigned) time(&t));

//Int_t n;
//printf("GIVE THE NUMBER OF PASSING MUONS:  ");
//scanf("%d\n",&n);

// Initializing the Canvas
TCanvas *c1 = new TCanvas("c1");
c1->SetWindowSize (gClient->GetDisplayWidth(), gClient->GetDisplayHeight());
c1->RaiseWindow();
Double_t w = gPad->GetWw()*gPad->GetAbsWNDC();
Double_t h = gPad->GetWh()*gPad->GetAbsHNDC();
Double_t R = 60;
Int_t Chambers = 2, Levels = 3, Tubes = 16;
double pi = 2*acos(0.0); // calculating pi
Double_t d=2*R*cos(pi*30/180); // horizontal identation at level changing
Double_t l=2*R*sin(pi*30/180); // vertical distance between the centers of tubes at different levels
Double_t s = 2*(R+d); // vertical distance between chambers  
Double_t ymin = -(Levels-1)*d-(1+Tubes/1.5)*R-s;
Double_t ymax = (1+Tubes/6)*R;
Double_t ymean = (ymin+ymax)/2;
Double_t xmean = ((Tubes-1)*2*R -l)/2;
Double_t xmin = xmean-((ymax-ymin)*w/h)/2;
Double_t xmax = xmean+((ymax-ymin)*w/h)/2;
c1->SetFixedAspectRatio();
c1->Range(xmin,ymin,xmax,ymax);

// Show Intro
// showintro(c1, R,xmin,xmax,xmean,ymin,ymean);

c1->SetFillColor(kGray+1);

// ATLAS Detector Simulation

// Defining storage matrices and useful constants 
double tot_tubes = Chambers*Levels*Tubes;
double **tub_cent;
tub_cent = dmatrix(tot_tubes,2);
int muon, detectors, muonres_size;
std::vector<double> muon_results;
Int_t points = 0;

// Definitions for statistical analysis
double drift_vel = 0.03; // electron's average drift velocity in gas detector, units mm/ns
TH1D *h1 = new TH1D("h1","GAME STATS 1: Drift time histogram",1025,0,1025); // drift time histogram
h1->GetXaxis()->SetTitle("Drift time t (ns)");
h1->GetYaxis()->SetTitle("Counts");


printf("\nSTART OF SIMULATION\n");
printf("----------------------------------------------------------\n");
for(int i=0;i<3;i++){ // Simulation for 3 muon trajectories (change the upper bound of i in loop for different number of trejectories)
c1->Clear();   

muon = i+1;
TString num = TString::Format("Cosmic Muon %d",muon);
TLatex *shownum = new TLatex();
shownum->SetTextSize(0.025);
shownum->DrawLatex(xmin+1*R,ymax-R,num);
TString detect = TString::Format("ATLAS MUON DETECTOR CHAMBERS");
TLatex *showdetect = new TLatex();
showdetect->SetTextSize(0.03);
showdetect->SetTextColor(kYellow-7);
showdetect->DrawLatex(xmax-14*R,ymin+2.5*R,detect);


tub_cent = tubes(R,Chambers,Levels,Tubes,l,d,s,tub_cent);
c1->Update();
//gSystem->Sleep(1000);

muon_results = muon_traj(c1,pi,R,xmin,xmax,ymin,ymax,tub_cent,tot_tubes);
muonres_size = muon_results.size();
detectors = (int) muon_results[muonres_size-1];
//gSystem->Sleep(1000);
if(detectors==0){
   points = points - 200;
   printf("MUON %d NOT-DETECTED: -200 points !!!\n",i+1);
   /*TString detectstatus = TString::Format("NOT-DETECTED");
   TLatex *showstatus = new TLatex();
   showstatus->SetTextSize(0.06);
   showstatus->SetTextColor(kRed);
   showstatus->DrawLatex(xmean-4*R,ymean+1.5*R,detectstatus);
   c1->Update();*/
}
else{
   points = points + detectors*100;
   printf("MUON %d DETECTED by %d detectors: +%d points !!!\n",i+1,detectors,detectors*100);
   /*TString detectstatus = TString::Format("DETECTED");
   TLatex *showstatus = new TLatex();
   showstatus->SetTextSize(0.06);
   showstatus->SetTextColor(kGreen-1);
   showstatus->DrawLatex(xmean-4*R,ymean+1.5*R,detectstatus);
   c1->Update();*/
}

// Filling the drift time histogram
muon_results.pop_back();
for(int j=0;j<muonres_size-1;j++){
   h1->Fill(muon_results[j]/drift_vel);
}

gSystem->Sleep(100);
}
printf("----------------------------------------------------------\n");
printf("END OF SIMULATION\n\n");
printf("TOTAL POINTS: %d\n",points);
//gSystem->Sleep(1000);

// Show Outro
//showoutro(c1,R,xmin,xmax,xmean,ymin,ymean,points);
//gSystem->Sleep(3000);

// Showing stats
//TCanvas *c2 = new TCanvas("c2"); // Canvas for drift time histogram
//h1->Draw();


// Free memory of the tubes centers matrix
free_dmatrix(tub_cent,tot_tubes);

//c1->Close();

//Running the app
app->Run();

return 0;
}

// Random number generator function definition
double randf(double n_min, double n_max){
	
	return n_min + (double) (n_max-n_min)*(rand())/( RAND_MAX);
}

// Generate 2D dmatrix function definition
double **dmatrix(int nRows,int nCols){
// Dynamically allocate memory for the 2D array

double **matrix = (double **)malloc(nRows * sizeof(double *));
    for (int i = 0; i < nRows; ++i) {
        matrix[i] = (double *)malloc(nCols * sizeof(double));
    }
	
	return matrix;
}

// Free the memory of the 2D dmatrix
int free_dmatrix(double **matrix,int nRows){
   for(int i=0;i<nRows;i++){
      free(matrix[i]);
   }
   free(matrix);
   return 1;
}

// Showing intro function definition
void showintro(TCanvas *c, Double_t R, Double_t xmin, Double_t xmax,Double_t xmean,Double_t ymin,Double_t ymean){

Double_t x_start = xmin+R;
Double_t x_end = xmax-R;
Double_t x_step = (x_end-x_start)/17;
Int_t j;
c->SetFillColor(kGray+1);

for(int j=0;j<=16;j++){
   if(j==0){
   // Title
TString title = TString::Format("#Delta.#Epsilon.#Theta. 7-15 Sep 2024");
TLatex *showtitle = new TLatex();
showtitle->SetTextSize(0.1);
showtitle->SetTextColor(kRed);
showtitle->DrawLatex(xmean-10.35*R,ymean+7*R,title);

   // Subtitle
TString sub = TString::Format("CERN ATLAS EXPERIMENT");
TLatex *showsub = new TLatex();
showsub->SetTextSize(0.07);
showsub->SetTextColor(kBlack);
showsub->DrawLatex(xmean-10*R,ymean-2*R,sub);

  // Subtitle 2
TString sub2 = TString::Format("Cosmic Muon Trajectory and Detection Simulation");
TLatex *showsub2 = new TLatex();
showsub2->SetTextSize(0.04);
showsub2->SetTextColor(kBlack);
showsub2->DrawLatex(xmean-10.3*R,ymean-3*R,sub2);

// Subtitle 3
TString sub3 = TString::Format("ARISTOTLE UNIVERSITY OF THESSALONIKI");
TLatex *showsub3 = new TLatex();
showsub3->SetTextSize(0.03);
showsub3->SetTextColor(kBlack);
showsub3->DrawLatex(xmean-7*R,ymin+2.5*R,sub3);
   }

   // Theme
TString theme = TString::Format("Cosmic Timeline Loading");
TLatex *showtheme = new TLatex();
showtheme->SetTextSize(0.04);
showtheme->SetTextColor(kAzure);
showtheme->DrawLatex(xmean-5*R,ymean+4*R,theme);

  // Loading Bar 1
TString bars1 = TString::Format("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
TLatex *showbars1 = new TLatex();
showbars1->SetTextSize(0.03);
showbars1->SetTextColor(kAzure);
showbars1->DrawLatex(xmin,ymean+1*R,bars1);

  // Loading Bar 2
TString bars2 = TString::Format("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
TLatex *showbars2 = new TLatex();
showbars2->SetTextSize(0.03);
showbars2->SetTextColor(kAzure);
showbars2->DrawLatex(xmin,ymean+3*R,bars2);

  // Counting ages 
if(j%2==0 && j!=16){
TString count = TString::Format("%d",2024-10*(7-j/2));
TLatex *showcount = new TLatex();
showcount->SetTextSize(0.045);
showcount->SetTextColor(kYellow-7);
showcount->DrawLatex(x_start+j*x_step-R,ymean+1.8*R,count);
c->Update();
gSystem->Sleep(400);
}
else if(j==16){
TString count = TString::Format("TODAY");
TLatex *showcount = new TLatex();
showcount->SetTextSize(0.045);
showcount->SetTextColor(kGreen-1);
showcount->DrawLatex(x_start+j*x_step-R,ymean+1.8*R,count);
c->Update();
gSystem->Sleep(2000); 
}
else{
TString count = TString::Format(">>>");
TLatex *showcount = new TLatex();
showcount->SetTextSize(0.045);
showcount->SetTextColor(kYellow-7);
showcount->DrawLatex(x_start+j*x_step-R/1.5,ymean+1.8*R,count);
c->Update();
gSystem->Sleep(400);  
}
}
}

// Showing outro function definition
void showoutro(TCanvas *c, Double_t R,Double_t xmin, Double_t xmax,Double_t xmean,Double_t ymin,Double_t ymean,Int_t points){
Double_t x_start = xmin+R;
Double_t x_end = xmax-R;

c->Clear();
c->SetFillColor(kGray+1);

// Title
TString title = TString::Format("#Delta.#Epsilon.#Theta. 7-15 Sep 2024");
TLatex *showtitle = new TLatex();
showtitle->SetTextSize(0.1);
showtitle->SetTextColor(kRed);
showtitle->DrawLatex(xmean-10.35*R,ymean+7*R,title);

// Subtitle
TString sub = TString::Format("CERN 1954-2024");
TLatex *showsub = new TLatex();
showsub->SetTextSize(0.07);
showsub->SetTextColor(kBlack);
showsub->DrawLatex(xmean-5.8*R,ymean-7*R,sub);

// Subtitle 2
TString sub2 = TString::Format("ARISTOTLE UNIVERSITY OF THESSALONIKI");
TLatex *showsub2 = new TLatex();
showsub2->SetTextSize(0.03);
showsub2->SetTextColor(kBlack);
showsub2->DrawLatex(xmean-7*R,ymin+2.5*R,sub2);

// Theme
TString theme = TString::Format("FINAL SCORE:");
TLatex *showtheme = new TLatex();
showtheme->SetTextSize(0.04);
showtheme->SetTextColor(kAzure);
showtheme->DrawLatex(xmean-3.5*R,ymean+4*R,theme);

// Loading Bar 1
TString bars1 = TString::Format("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
TLatex *showbars1 = new TLatex();
showbars1->SetTextSize(0.03);
showbars1->SetTextColor(kAzure);
showbars1->DrawLatex(xmin,ymean-R,bars1);

  // Loading Bar 2
TString bars2 = TString::Format("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
TLatex *showbars2 = new TLatex();
showbars2->SetTextSize(0.03);
showbars2->SetTextColor(kAzure);
showbars2->DrawLatex(xmin,ymean+3*R,bars2);
c->Update();
gSystem->Sleep(2000);


  // End Message 
TString endtxt = TString::Format("%d pts",points);
TLatex *showend = new TLatex();
showend->SetTextSize(0.07);
showend->SetTextColor(kYellow-7);
showend->DrawLatex(xmean-3.5*R,ymean+0.5*R,endtxt);
c->Update();
gSystem->Sleep(2000);

// End Message 2
TString endtxt2 = TString::Format("HAPPY 70th CERN FOUNDING ANNIVERSARY !!!");
TLatex *showend2 = new TLatex();
showend2->SetTextSize(0.05);
showend2->SetTextColor(kBlue);
showend2->DrawLatex(xmean-13*R,ymean-2*R,endtxt2);
c->Update();
}

// Tubes drawing function definition
double **tubes(double R,int Chamb,int Lvs,int Tubes,double l,double d, double s,double **tubes_centers)
{
double x_cent, y_cent;
int n=0;
for(Int_t k=0;k<Chamb;k++){ 
for(Int_t i=0;i<Lvs;i++){
for(Int_t j=0;j<Tubes;j++)
{  x_cent = j*2*R - (i%2)*l, y_cent = -i*d - s*2*k;
   tubes_centers[n][0] = x_cent;
   tubes_centers[n][1] = y_cent;
   TEllipse *el1 = new TEllipse(x_cent,y_cent,R,R);
   el1->SetLineWidth(2);
   el1->SetFillColor(kGray);
   el1->Draw("SAME");
   TMarker *m1 = new TMarker(x_cent,y_cent,8);
   m1->SetMarkerColorAlpha(kRed-6, 0.20);
   m1->Draw("SAME");
   n = n + 1;
}
}
}
return tubes_centers;
}


// Muon trajectory and detection tubes drawing function definition
std::vector<double> muon_traj(TCanvas *c,double pi,double R,double xmin, double xmax,double ymin, double ymax,double **tubes_centers,double tot_tubes){

Int_t n = 500;
Double_t angl = randf(70,110);
int tot_detect;

vector<double> det_tubes_X,det_tubes_Y,det_tubes_d;

Double_t x_m = (xmax+xmin)/2;   
Double_t x_rand = randf(xmin+3*R,xmax-3*R); 
Double_t slope = tan(angl*pi/180);
Double_t y0 = randf(ymin+2*R,ymax-2*R);
Double_t intercept = y0 - slope*x_rand;
Double_t y_line_min = slope*xmin + intercept, y_line_max = slope*xmax + intercept;

double x_cent;
double y_cent;
double xline;
double yline;
double xstep;
double r, d;

for(int i=0;i<tot_tubes;i++){
x_cent = tubes_centers[i][0];
y_cent = tubes_centers[i][1];

xstep = 2*R/n;

for(int j=0;j<n;j++){
xline = x_cent-R+j*xstep;
yline = slope*xline + intercept;

r = sqrt(pow(xline - x_cent,2)+pow(yline-y_cent,2));
if(r<=R){
   det_tubes_X.push_back(x_cent);
   det_tubes_Y.push_back(y_cent);
   d = abs(-slope*x_cent + y_cent - intercept)/sqrt(pow(slope,2)+1);
   det_tubes_d.push_back(d);
   break;
}
}
}

tot_detect = det_tubes_X.size();

for(int i=0;i<tot_detect;i++){
TEllipse *el2 = new TEllipse(det_tubes_X[i],det_tubes_Y[i],R,R);
el2->SetLineWidth(2);
//el2->SetLineColor(kGreen);
//el2->SetFillStyle(0);
el2->SetFillColor(kGreen);
el2->Draw("SAME");
TMarker *m2 = new TMarker(det_tubes_X[i],det_tubes_Y[i],8);
m2->SetMarkerColorAlpha(kRed-6, 0.20);
//m2->SetMarkerSize(1.5);
m2->Draw("SAME");
c->Update();
gSystem->Sleep(300);
}

for(int i=0;i<tot_detect;i++){
TEllipse *el3 = new TEllipse(det_tubes_X[i],det_tubes_Y[i],det_tubes_d[i],det_tubes_d[i]);
el3->SetLineWidth(2);
el3->SetLineColor(kAzure);
el3->SetFillStyle(0);
//el2->SetFillColor(kGreen);
el3->Draw("SAME");
}
c->Update();

gSystem->Sleep(1000);
double x_step = (xmax-xmin)/n;

for(int i=0;i<n;i++){
if(slope>=0){
Double_t xlow = xmax-(i+1)*x_step, xup = xmax-i*x_step; 
Double_t ylow = slope*xlow + intercept, yup = slope*xup + intercept;

TLine *traj = new TLine(xlow,ylow,xup,yup);
if(ylow>ymax+2*R){continue;}
else if(yup<ymin){break;}
else{
traj->SetLineColor(kRed);
traj->SetLineWidth(3);
traj->Draw("SAME");
c->Update();

gSystem->Sleep(10*pow(abs(slope),0.25));
}
}   
else if(slope<0){
Double_t xlow = xmin+i*x_step, xup = xmin+(i+1)*x_step; 
Double_t ylow = slope*xlow + intercept, yup = slope*xup+intercept;

TLine *traj = new TLine(xlow,ylow,xup,yup);
if(ylow>ymax+2*R){continue;}
else if(yup<ymin){break;}
else{
traj->SetLineColor(kRed);
traj->SetLineWidth(3);
traj->Draw("SAME");
c->Update();

gSystem->Sleep(10*pow(abs(slope),0.25));
}
}
}
det_tubes_d.push_back((double) tot_detect);

return det_tubes_d;
}