from app.core.config import Settings, get_settings

class PortfolioService :
    def __init__(self, config:Settings) -> None:
        pass

portfolio_service = PortfolioService(config=get_settings())

def get_portfolio_service() :
    yield portfolio_service

