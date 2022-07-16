__kernel void gaussian_map(
    read_only image2d_t XYZR,
    __global const float* params,
    write_only image3d_t Grid) {
  const int ii = get_global_id(0);
  const int jj = get_global_id(1);
  const int kk = get_global_id(2);

  float angs_per_voxel = (float)(params[0]);
  float dist = (float)(params[1]);
  float inner_intensity = (float)(params[2]);
  float dist_sq = pown(dist, 2);
  float3 centroid = (float3)(get_image_width(Grid)/2+0.5, get_image_height(Grid)/2+0.5, get_image_depth(Grid)/2+0.5);

  float3 cent = (float3)(ii+0.5, jj+0.5, kk+0.5);
  float3 uvw = (float3)(-1*angs_per_voxel*(centroid.x-cent.x) , -1*angs_per_voxel*(centroid.y-cent.y) , -1*angs_per_voxel*(centroid.z-cent.z));
  float intensity = 0;
  float fi = 0;

  //for x, y, z, sigma in zip(X, Y, Z, atom_radius) {
  for (int width = 0; width < get_image_width(XYZR); ++width) {
    float x = read_imagef(XYZR, (int2)(width, 0)).x;
    float y = read_imagef(XYZR, (int2)(width, 1)).x;
    float z = read_imagef(XYZR, (int2)(width, 2)).x;
    float sigma = read_imagef(XYZR, (int2)(width, 3)).x;

    float cst = 1. / (sqrt(2 * M_PI) * sigma);

    float r_sq = pown((uvw.x-x), 2) + pown((uvw.y-y), 2) + pown((uvw.z-z), 2);

    if (r_sq <= dist_sq) {
      float I = cst * exp(-0.5 * pown((sqrt(r_sq) / sigma), 2));
      intensity += I;
    }
  }
  if (intensity >= 0.2) {
    fi = inner_intensity;
  }

  write_imagef(Grid, (int4)(kk, jj, ii, 0), (float4)(fi, 0, 0, 0));
}

__kernel void quantize(
    read_only image3d_t Map,
    __global const float* params,
    write_only image3d_t Grid) {
  const int ii = get_global_id(0);
  const int jj = get_global_id(1);
  const int kk = get_global_id(2);
  float inner_intensity = (float)(params[2]);
  float surface_intensity = (float)(params[3]);

  float neighbor_sum = 0;
  float val = 0;

  int upper = get_image_width(Map);

  for (int xx = max(0, kk-1); xx < min(upper, kk+2); ++xx) {
      for (int yy = max(0, jj-1); yy < min(upper, jj+2); ++yy) {
      	  for (int zz = max(0, ii-1); zz < min(upper, ii+2); ++zz) {
	      neighbor_sum += read_imagef(Map, (int4)(xx, yy, zz, 0)).x;
	  }
      }
  }

  if (neighbor_sum != 0) {
     if (fabs(neighbor_sum) >= fabs(26*inner_intensity)) {
     	val = inner_intensity;
     } else if (0 < fabs(neighbor_sum) < fabs(26*inner_intensity)) {
       val = surface_intensity;
     }
  }

  write_imagef(Grid, (int4)(kk, jj, ii, 0), (float4)(val, 0, 0, 0));
}
