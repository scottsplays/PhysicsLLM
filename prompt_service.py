"""Entry point kept so `streamlit run prompt_service.py` opens PhysicsSLM.

The handwritten library is unchanged. This file does not import dataloader.py.
"""

from physics_slm_app import main

if __name__ == "__main__":
    main()
