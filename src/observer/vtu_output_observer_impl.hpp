#pragma once
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkHigherOrderQuadrilateral.h>
#include <vtkLagrangeQuadrilateral.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

namespace DGSEM {
namespace detail {

std::vector<int> vtk_lagrange_order(int NNodes) {
  int N = NNodes - 1;
  std::vector<int> map;

  // -----------------
  // 1. corners
  // -----------------
  map.push_back(0);              // (0,0)
  map.push_back(N);              // (N,0)
  map.push_back(N * NNodes + N); // (N,N)
  map.push_back(N * NNodes);     // (0,N)

  // -----------------
  // 2. bottom edge
  // -----------------
  for (int i = 1; i < N; ++i)
    map.push_back(i);

  // -----------------
  // 3. right edge
  // -----------------
  for (int j = 1; j < N; ++j)
    map.push_back(j * NNodes + N);

  // -----------------
  // 4. top edge
  // -----------------
  for (int i = N - 1; i > 0; --i)
    map.push_back(N * NNodes + i);

  // -----------------
  // 5. left edge
  // -----------------
  for (int j = N - 1; j > 0; --j)
    map.push_back(j * NNodes);

  // -----------------
  // 6. interior
  // -----------------
  for (int j = 1; j < N; ++j)
    for (int i = 1; i < N; ++i)
      map.push_back(j * NNodes + i);

  return map;
}

int vtk_index(int i, int j, int N) {
  int NNodes = N + 1;

  if (i == 0 && j == 0)
    return 0;
  if (i == N && j == 0)
    return 1;
  if (i == N && j == N)
    return 2;
  if (i == 0 && j == N)
    return 3;

  int id = 4;

  if (j == 0)
    return id + (i - 1);
  id += (N - 1);

  if (i == N)
    return id + (j - 1);
  id += (N - 1);

  if (j == N)
    return id + (N - 1 - (i - 1));
  id += (N - 1);

  if (i == 0)
    return id + (N - 1 - (j - 1));
  id += (N - 1);

  return id + (j - 1) * (N - 1) + (i - 1);
}

inline void write_vtu_lagrange_quad(const std::string& filename,
                                    const std::vector<double>& points,
                                    const std::vector<double>& values, int nx,
                                    int ny, int NNodes, int nvars) {
  const int ndof = NNodes * NNodes;
  const int nelem = nx * ny;
  const int total_points = nelem * ndof;

  // ----------------------------
  // points
  // ----------------------------
  auto vtk_points = vtkSmartPointer<vtkPoints>::New();
  vtk_points->SetNumberOfPoints(total_points);

  for (int i = 0; i < total_points; ++i)
    vtk_points->SetPoint(i, points[3 * i + 0], points[3 * i + 1],
                         points[3 * i + 2]);

  auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  grid->SetPoints(vtk_points);

  // ----------------------------
  // cells（每个DG单元一个cell）
  // ----------------------------
  int gid = 0;

  auto reorder = vtk_lagrange_order(NNodes);

  int N = NNodes - 1;

  for (int e = 0; e < nelem; ++e) {
    auto cell = vtkSmartPointer<vtkLagrangeQuadrilateral>::New();
    cell->GetPointIds()->SetNumberOfIds(ndof);

    for (int d = 0; d < ndof; ++d) {
      cell->GetPointIds()->SetId(d, e * ndof + d);
    }

    cell->SetOrder(NNodes - 1, NNodes - 1);

    grid->InsertNextCell(cell->GetCellType(), cell->GetPointIds());
  }
  // ----------------------------
  // data
  // ----------------------------
  for (int v = 0; v < nvars; ++v) {
    auto arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(("var" + std::to_string(v)).c_str());
    arr->SetNumberOfTuples(total_points);

    for (int i = 0; i < total_points; ++i)
      arr->SetValue(i, values[i * nvars + v]);

    grid->GetPointData()->AddArray(arr);
  }

  // ----------------------------
  // writer（binary）
  // ----------------------------
  auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData(grid);

  writer->SetDataModeToAppended();
  writer->EncodeAppendedDataOff(); // raw binary

  writer->Write();
}

} // namespace detail
} // namespace DGSEM