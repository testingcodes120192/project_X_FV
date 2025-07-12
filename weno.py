# weno.py
import numpy as np

class WENOReconstructor:
    """
    WENO (Weighted Essentially Non-Oscillatory) reconstruction for finite volume methods.
    
    This class implements 5th order WENO reconstruction with options for 
    lower order schemes (1st and 2nd order) for comparison and efficiency.
    """
    
    def __init__(self, order=5, eps=1e-6):
        """
        Initialize WENO reconstructor.
        
        Parameters
        ----------
        order : int
            Spatial order of accuracy (1, 2, or 5)
        eps : float
            Small parameter to avoid division by zero in weights
        """
        self.order = order
        self.eps = eps
        
        # WENO5 optimal weights
        self.gamma = {
            'right': np.array([0.3, 0.6, 0.1]),  # Positive bias
            'left': np.array([0.1, 0.6, 0.3])     # Negative bias
        }
        
    def reconstruct_x(self, field, j, order=None):
        """
        Reconstruct values at x-direction cell faces for row j.
        
        Parameters
        ----------
        field : ndarray
            2D field array with ghost cells
        j : int
            Row index
        order : int, optional
            Override default order for this reconstruction
            
        Returns
        -------
        face_values : ndarray
            Values at cell faces (size nx+1 for interior)
        """
        if order is None:
            order = self.order
            
        if order == 1:
            return self._reconstruct_1st_order_x(field, j)
        elif order == 2:
            return self._reconstruct_2nd_order_x(field, j)
        elif order == 5:
            return self._reconstruct_weno5_x(field, j)
        else:
            raise ValueError(f"Unsupported order: {order}")
    
    def reconstruct_y(self, field, i, order=None):
        """
        Reconstruct values at y-direction cell faces for column i.
        
        Parameters
        ----------
        field : ndarray
            2D field array with ghost cells
        i : int
            Column index
        order : int, optional
            Override default order for this reconstruction
            
        Returns
        -------
        face_values : ndarray
            Values at cell faces (size ny+1 for interior)
        """
        if order is None:
            order = self.order
            
        if order == 1:
            return self._reconstruct_1st_order_y(field, i)
        elif order == 2:
            return self._reconstruct_2nd_order_y(field, i)
        elif order == 5:
            return self._reconstruct_weno5_y(field, i)
        else:
            raise ValueError(f"Unsupported order: {order}")
    
    def _reconstruct_1st_order_x(self, field, j):
        """First order reconstruction (piecewise constant)."""
        # For first order, face value is just the upwind cell value
        # For diffusion, we'll use simple average
        return 0.5 * (field[j, :-1] + field[j, 1:])
    
    def _reconstruct_1st_order_y(self, field, i):
        """First order reconstruction (piecewise constant)."""
        return 0.5 * (field[:-1, i] + field[1:, i])
    
    def _reconstruct_2nd_order_x(self, field, j):
        """Second order reconstruction (linear)."""
        # Linear reconstruction using cell averages
        # This gives 2nd order accuracy for smooth solutions
        n = len(field[j, :])
        face_values = np.zeros(n-1)
        
        for i in range(1, n-2):
            # Simple centered average
            face_values[i] = 0.5 * (field[j, i] + field[j, i+1])
            
        # Boundary faces
        face_values[0] = 0.5 * (field[j, 0] + field[j, 1])
        face_values[-1] = 0.5 * (field[j, -2] + field[j, -1])
        
        return face_values
    
    def _reconstruct_2nd_order_y(self, field, i):
        """Second order reconstruction (linear)."""
        n = len(field[:, i])
        face_values = np.zeros(n-1)
        
        for j in range(1, n-2):
            face_values[j] = 0.5 * (field[j, i] + field[j+1, i])
            
        face_values[0] = 0.5 * (field[0, i] + field[1, i])
        face_values[-1] = 0.5 * (field[-2, i] + field[-1, i])
        
        return face_values
    
    def _reconstruct_weno5_x(self, field, j):
        """Fifth order WENO reconstruction in x-direction."""
        n = len(field[j, :])
        face_values = np.zeros(n-1)
        
        # Need at least 6 cells for WENO5 stencil
        for i in range(3, n-3):
            # Get stencil values
            v = field[j, i-2:i+4]  # 6 values: v[0] to v[5]
            
            # Reconstruct from left (positive bias) and right (negative bias)
            u_left = self._weno5_positive(v[:-1])   # Use v[0:5]
            u_right = self._weno5_negative(v[1:])   # Use v[1:6]
            
            # For diffusion, average the reconstructions
            face_values[i] = 0.5 * (u_left + u_right)
        
        # Fill boundary faces with 2nd order
        for i in range(3):
            face_values[i] = 0.5 * (field[j, i] + field[j, i+1])
        for i in range(n-3, n-1):
            face_values[i] = 0.5 * (field[j, i] + field[j, i+1])
            
        return face_values
    
    def _reconstruct_weno5_y(self, field, i):
        """Fifth order WENO reconstruction in y-direction."""
        n = len(field[:, i])
        face_values = np.zeros(n-1)
        
        for j in range(3, n-3):
            v = field[j-2:j+4, i]
            
            u_left = self._weno5_positive(v[:-1])
            u_right = self._weno5_negative(v[1:])
            
            face_values[j] = 0.5 * (u_left + u_right)
        
        # Boundary faces
        for j in range(3):
            face_values[j] = 0.5 * (field[j, i] + field[j+1, i])
        for j in range(n-3, n-1):
            face_values[j] = 0.5 * (field[j, i] + field[j+1, i])
            
        return face_values
    
    def _weno5_positive(self, v):
        """
        WENO5 reconstruction with positive bias.
        
        Parameters
        ----------
        v : ndarray
            5 cell values [v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}]
            
        Returns
        -------
        float
            Reconstructed value at interface i+1/2
        """
        # Three candidate stencils
        # S0: {i-2, i-1, i}
        # S1: {i-1, i, i+1}
        # S2: {i, i+1, i+2}
        
        # Candidate values (3rd order polynomials evaluated at i+1/2)
        u = np.zeros(3)
        u[0] = (2*v[0] - 7*v[1] + 11*v[2]) / 6.0
        u[1] = (-v[1] + 5*v[2] + 2*v[3]) / 6.0
        u[2] = (2*v[2] + 5*v[3] - v[4]) / 6.0
        
        # Smoothness indicators
        IS = np.zeros(3)
        IS[0] = 13.0/12.0 * (v[0] - 2*v[1] + v[2])**2 + \
                0.25 * (v[0] - 4*v[1] + 3*v[2])**2
        IS[1] = 13.0/12.0 * (v[1] - 2*v[2] + v[3])**2 + \
                0.25 * (v[1] - v[3])**2
        IS[2] = 13.0/12.0 * (v[2] - 2*v[3] + v[4])**2 + \
                0.25 * (3*v[2] - 4*v[3] + v[4])**2
        
        # Nonlinear weights
        alpha = self.gamma['right'] / (self.eps + IS)**2
        w = alpha / np.sum(alpha)
        
        # WENO reconstruction
        return np.sum(w * u)
    
    def _weno5_negative(self, v):
        """
        WENO5 reconstruction with negative bias.
        
        Parameters
        ----------
        v : ndarray
            5 cell values [v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}]
            
        Returns
        -------
        float
            Reconstructed value at interface i-1/2
        """
        # Mirror the stencil and use positive reconstruction
        v_mirror = v[::-1].copy()
        return self._weno5_positive(v_mirror)
    
    def compute_derivatives_x(self, field, mesh, order=None):
        """
        Compute x-derivatives at cell centers using reconstruction.
        
        Parameters
        ----------
        field : ndarray
            2D field array with ghost cells
        mesh : FVMesh
            Mesh object
        order : int, optional
            Order of accuracy
            
        Returns
        -------
        ndarray
            x-derivative at cell centers (interior only)
        """
        if order is None:
            order = self.order
            
        j_int, i_int = mesh.get_interior_slice()
        #ny_int = j_int.stop - j_int.start
        #nx_int = i_int.stop - i_int.start
        
        ny_int = mesh.ny
        nx_int = mesh.nx
        
        dfdx = np.zeros((ny_int, nx_int))
        
        for j_local, j_global in enumerate(range(j_int.start, j_int.stop)):
            # Reconstruct at faces
            face_values = self.reconstruct_x(field, j_global, order)
            
            # Compute derivative from face values
            i_start = i_int.start
            for i_local in range(nx_int):
                dfdx[j_local, i_local] = (face_values[i_start + i_local] - 
                                          face_values[i_start + i_local - 1]) / mesh.dx
                
        return dfdx
    
    def compute_derivatives_y(self, field, mesh, order=None):
        """
        Compute y-derivatives at cell centers using reconstruction.
        
        Parameters
        ----------
        field : ndarray
            2D field array with ghost cells
        mesh : FVMesh
            Mesh object
        order : int, optional
            Order of accuracy
            
        Returns
        -------
        ndarray
            y-derivative at cell centers (interior only)
        """
        if order is None:
            order = self.order
            
        j_int, i_int = mesh.get_interior_slice()
        #ny_int = j_int.stop - j_int.start
        #nx_int = i_int.stop - i_int.start
        
        ny_int = mesh.ny
        nx_int = mesh.nx
        
        dfdy = np.zeros((ny_int, nx_int))
        
        for i_local, i_global in enumerate(range(i_int.start, i_int.stop)):
            face_values = self.reconstruct_y(field, i_global, order)
            
            j_start = j_int.start
            for j_local in range(ny_int):
                dfdy[j_local, i_local] = (face_values[j_start + j_local] - 
                                          face_values[j_start + j_local - 1]) / mesh.dy
                
        return dfdy
    
    def compute_laplacian(self, field, mesh, alpha, order=None):
        """
        Compute Laplacian for diffusion equation using finite volume method.
        
        For heat equation: ∂T/∂t = α∇²T
        We compute -∇·(-α∇T) = α∇²T
        """
        if order is None:
            order = self.order
            
        j_int, i_int = mesh.get_interior_slice()
        ny_int = mesh.ny
        nx_int = mesh.nx
        laplacian = np.zeros((ny_int, nx_int))
        
        # For low-order methods, use simple finite differences
        if order <= 2:
            # Simple 2nd order centered differences
            g = mesh.ghost_cells
            for j_local in range(ny_int):
                j = j_local + g
                for i_local in range(nx_int):
                    i = i_local + g
                    
                    # ∂²T/∂x²
                    d2Tdx2 = (field[j, i+1] - 2*field[j, i] + field[j, i-1]) / mesh.dx**2
                    
                    # ∂²T/∂y²
                    d2Tdy2 = (field[j+1, i] - 2*field[j, i] + field[j-1, i]) / mesh.dy**2
                    
                    laplacian[j_local, i_local] = alpha * (d2Tdx2 + d2Tdy2)
                    
        else:
            # For WENO5, compute fluxes at faces
            g = mesh.ghost_cells
            
            # X-direction diffusive flux: F = -α ∂T/∂x
            for j_local in range(ny_int):
                j = j_local + g
                
                # Compute gradients at faces using centered differences
                # This is more stable for diffusion than using WENO reconstruction
                flux_x = np.zeros(nx_int + 1)
                for i_face in range(nx_int + 1):
                    i = i_face + g
                    # Gradient at face between cells i-1 and i
                    dTdx = (field[j, i] - field[j, i-1]) / mesh.dx
                    flux_x[i_face] = -alpha * dTdx
                
                # Flux divergence
                for i_local in range(nx_int):
                    laplacian[j_local, i_local] += -(flux_x[i_local + 1] - flux_x[i_local]) / mesh.dx
            
            # Y-direction diffusive flux: G = -α ∂T/∂y
            for i_local in range(nx_int):
                i = i_local + g
                
                # Compute gradients at faces
                flux_y = np.zeros(ny_int + 1)
                for j_face in range(ny_int + 1):
                    j = j_face + g
                    # Gradient at face between cells j-1 and j
                    dTdy = (field[j, i] - field[j-1, i]) / mesh.dy
                    flux_y[j_face] = -alpha * dTdy
                
                # Flux divergence
                for j_local in range(ny_int):
                    laplacian[j_local, i_local] += -(flux_y[j_local + 1] - flux_y[j_local]) / mesh.dy
        
        return laplacian