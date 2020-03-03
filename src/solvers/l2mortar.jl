module L2Mortar

using ..Interpolation: gauss_lobatto_nodes_weights, barycentric_weights,
                       lagrange_interpolating_polynomials, gauss_nodes_weights,
                       polynomial_interpolation_matrix


function calc_forward_upper(n_nodes)
  # Calculate nodes, weights, and barycentric weights
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  wbary = barycentric_weights(nodes)

  # Calculate projection matrix (actually: interpolation)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] + 1), nodes, wbary)
    for i in 1:n_nodes
      operator[j, i] = poly[i]
    end
  end

  return operator
end


function calc_forward_lower(n_nodes)
  # Calculate nodes, weights, and barycentric weights
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  wbary = barycentric_weights(nodes)

  # Calculate projection matrix (actually: interpolation)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] - 1), nodes, wbary)
    for i in 1:n_nodes
      operator[j, i] = poly[i]
    end
  end

  return operator
end


function calc_reverse_upper(n_nodes)
  # Calculate nodes, weights, and barycentric weights for Legendre-Gauss
  gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes)
  gauss_wbary = barycentric_weights(gauss_nodes)

  # Calculate projection matrix (actually: discrete L2 projection with errors)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (gauss_nodes[j] + 1), gauss_nodes, gauss_wbary)
    for i in 1:n_nodes
      operator[i, j] = 1/2 * poly[i] * gauss_weights[j]/gauss_weights[i]
    end
  end

  # Calculate Vandermondes
  lobatto_nodes, lobatto_weights = gauss_lobatto_nodes_weights(n_nodes)
  gauss2lobatto = polynomial_interpolation_matrix(gauss_nodes, lobatto_nodes)
  lobatto2gauss = polynomial_interpolation_matrix(lobatto_nodes, gauss_nodes)

  return gauss2lobatto * operator * lobatto2gauss
end


function calc_reverse_lower(n_nodes)
  # Calculate nodes, weights, and barycentric weights for Legendre-Gauss
  gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes)
  gauss_wbary = barycentric_weights(gauss_nodes)

  # Calculate projection matrix (actually: discrete L2 projection with errors)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (gauss_nodes[j] - 1), gauss_nodes, gauss_wbary)
    for i in 1:n_nodes
      operator[i, j] = 1/2 * poly[i] * gauss_weights[j]/gauss_weights[i]
    end
  end

  # Calculate Vandermondes
  lobatto_nodes, lobatto_weights = gauss_lobatto_nodes_weights(n_nodes)
  gauss2lobatto = polynomial_interpolation_matrix(gauss_nodes, lobatto_nodes)
  lobatto2gauss = polynomial_interpolation_matrix(lobatto_nodes, gauss_nodes)

  return gauss2lobatto * operator * lobatto2gauss
end


end # module Mortar
