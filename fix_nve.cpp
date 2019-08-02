/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cstdio>
#include <cstring>
#include "fix_nve.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVE::FixNVE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"nve/sphere") != 0 && narg < 3)
    error->all(FLERR,"Illegal fix nve command");

  dynamic_group_allow = 1;
  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixNVE::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVE::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  B = 1.0;									//edited

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVE::initial_integrate(int /*vflag*/)
{
  double dtfm;
  double qB,dtfmqB;								//edited
  double vx,vy;									//edited

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *q = atom->q;							//edited
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
		qB = q[i]*B;							//edited
		dtfmqB = dtfm * qB						//edited
		vx = v[i][0];							//edited
		vy = v[i][1];							//edited
		v[i][0] += dtfm * (f[i][0] + qB * vy);			//edited
		v[i][1] += dtfm * (f[i][1] + qB * vx);			//edited
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
		v[i][0] += dtfmqB * (vy + dtfm * f[i][1] - dtfmqB * 2.0 * vx);			//edited
		v[i][1] += dtfmqB * (vx - dtfm * f[i][0] - dtfmqB * 2.0 * vy);			//edited
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
		qB = q[i]*B;							//edited
		dtfmqB = dtfm * qB						//edited
		vx = v[i][0];							//edited
		vy = v[i][1];							//edited
		v[i][0] += dtfm * (f[i][0] + qB * vy);			//edited
		v[i][1] += dtfm * (f[i][1] + qB * vx);			//edited
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
		v[i][0] += dtfmqB * (vy + dtfm * f[i][1] - dtfmqB * 2.0 * vx);			//edited
		v[i][1] += dtfmqB * (vx - dtfm * f[i][0] - dtfmqB * 2.0 * vy);			//edited
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate()
{
  double dtfm;
  double dtfmqB;							//edited

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *q = atom->q;							//edited
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
		dtfmqB = dtfm * q[i] * B;									//edited
		v[i][0] += dtfm * (f[i][0] + dtfmqB * f[i][1]);				//edited
		v[i][1] += dtfm * (f[i][1] - dtfmqB * f[i][0]);				//edited
        v[i][2] += dtfm * f[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
		dtfmqB = dtfm * q[i] * B;									//edited
		v[i][0] += dtfm * (f[i][0] + dtfmqB * f[i][1]);				//edited
		v[i][1] += dtfm * (f[i][1] - dtfmqB * f[i][0]);				//edited
        v[i][2] += dtfm * f[i][2];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVE::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
