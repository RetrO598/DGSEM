#pragma once
#include <string>
#include <vector>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkHigherOrderQuadrilateral.h>
#include <vtkLagrangeHexahedron.h>
#include <vtkLagrangeQuadrilateral.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

namespace DGSEM {
namespace detail {

inline void write_vtu_lagrange_quad(const std::string& filename,
                                    const std::vector<double>& points,
                                    const std::vector<double>& values,
                                    const std::vector<std::string>& var_names,
                                    int nx, int ny, int NNodes, int nvars) {
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
    arr->SetName(var_names[v].c_str());
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

inline void write_vtu_lagrange_hex(const std::string& filename,
                                   const std::vector<double>& points,
                                   const std::vector<double>& values,
                                   const std::vector<std::string>& var_names,
                                   int nx, int ny, int nz, int NNodes,
                                   int nvars) {
  const int ndof = NNodes * NNodes * NNodes;
  const int nelem = nx * ny * nz;
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
  for (int e = 0; e < nelem; ++e) {
    auto cell = vtkSmartPointer<vtkLagrangeHexahedron>::New();
    cell->GetPointIds()->SetNumberOfIds(ndof);
    for (int d = 0; d < ndof; ++d) {
      cell->GetPointIds()->SetId(d, e * ndof + d);
    }
    cell->SetOrder(NNodes - 1, NNodes - 1, NNodes - 1);
    grid->InsertNextCell(cell->GetCellType(), cell->GetPointIds());
  }

  // ----------------------------
  // data
  // ----------------------------
  for (int v = 0; v < nvars; ++v) {
    auto arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(var_names[v].c_str());
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