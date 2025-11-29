"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
import math
from langchain_core.tools import tool
import rag

logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)


@tool
def calculate_loan(amount: float, interest_rate: float, term_months: int, payment_type: str = "annuity") -> str:
    """
    Рассчитывает параметры кредита: ежемесячный платеж, переплату и общую сумму.
    
    Args:
        amount: Сумма кредита в рублях
        interest_rate: Процентная ставка годовых (например, 12.5 для 12.5%)
        term_months: Срок кредита в месяцах
        payment_type: Тип платежа - "annuity" (аннуитетный) или "differentiated" (дифференцированный)
    
    Returns:
        JSON с расчетом: monthly_payment, total_payment, overpayment, payment_schedule (первые 6 месяцев)
    """
    try:
        monthly_rate = interest_rate / 100 / 12
        
        if payment_type == "annuity":
            # Аннуитетный платеж
            if monthly_rate == 0:
                monthly_payment = amount / term_months
            else:
                monthly_payment = amount * (monthly_rate * (1 + monthly_rate) ** term_months) / \
                                ((1 + monthly_rate) ** term_months - 1)
            total_payment = monthly_payment * term_months
            overpayment = total_payment - amount
            
            # График платежей (первые 6 месяцев)
            payment_schedule = []
            remaining_balance = amount
            for month in range(1, min(7, term_months + 1)):
                interest_part = remaining_balance * monthly_rate
                principal_part = monthly_payment - interest_part
                remaining_balance -= principal_part
                payment_schedule.append({
                    "month": month,
                    "payment": round(monthly_payment, 2),
                    "principal": round(principal_part, 2),
                    "interest": round(interest_part, 2),
                    "remaining_balance": round(max(0, remaining_balance), 2)
                })
        else:
            # Дифференцированный платеж
            principal_part = amount / term_months
            total_payment = 0
            payment_schedule = []
            remaining_balance = amount
            
            for month in range(1, min(7, term_months + 1)):
                interest_part = remaining_balance * monthly_rate
                monthly_payment = principal_part + interest_part
                remaining_balance -= principal_part
                total_payment += monthly_payment
                payment_schedule.append({
                    "month": month,
                    "payment": round(monthly_payment, 2),
                    "principal": round(principal_part, 2),
                    "interest": round(interest_part, 2),
                    "remaining_balance": round(max(0, remaining_balance), 2)
                })
            
            # Для дифференцированного платежа первый платеж - максимальный
            monthly_payment = payment_schedule[0]["payment"] if payment_schedule else 0
            overpayment = total_payment - amount
        
        result = {
            "amount": round(amount, 2),
            "interest_rate": interest_rate,
            "term_months": term_months,
            "payment_type": payment_type,
            "monthly_payment": round(monthly_payment, 2),
            "total_payment": round(total_payment, 2),
            "overpayment": round(overpayment, 2),
            "overpayment_percent": round((overpayment / amount) * 100, 2),
            "payment_schedule": payment_schedule
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in calculate_loan: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def calculate_deposit(amount: float, interest_rate: float, term_months: int, capitalization: bool = True) -> str:
    """
    Рассчитывает доходность вклада с учетом капитализации процентов.
    
    Args:
        amount: Сумма вклада в рублях
        interest_rate: Процентная ставка годовых (например, 8.5 для 8.5%)
        term_months: Срок вклада в месяцах
        capitalization: Капитализация процентов (True) или без капитализации (False)
    
    Returns:
        JSON с расчетом: final_amount, interest_earned, effective_rate
    """
    try:
        annual_rate = interest_rate / 100
        
        if capitalization:
            # С капитализацией (сложный процент)
            # Конвертируем месяцы в годы для расчета
            years = term_months / 12
            final_amount = amount * (1 + annual_rate) ** years
            interest_earned = final_amount - amount
            # Эффективная ставка с учетом капитализации
            effective_rate = ((final_amount / amount) ** (1 / years) - 1) * 100
        else:
            # Без капитализации (простые проценты)
            years = term_months / 12
            interest_earned = amount * annual_rate * years
            final_amount = amount + interest_earned
            effective_rate = interest_rate
        
        result = {
            "initial_amount": round(amount, 2),
            "interest_rate": interest_rate,
            "term_months": term_months,
            "capitalization": capitalization,
            "final_amount": round(final_amount, 2),
            "interest_earned": round(interest_earned, 2),
            "effective_rate": round(effective_rate, 2),
            "monthly_interest": round(interest_earned / term_months, 2)
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in calculate_deposit: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует валюту по курсу Сбербанка.
    
    Args:
        amount: Сумма для конвертации
        from_currency: Исходная валюта (RUB, USD, EUR, GBP, CNY, JPY)
        to_currency: Целевая валюта (RUB, USD, EUR, GBP, CNY, JPY)
    
    Returns:
        JSON с результатом конвертации и курсом
    """
    try:
        # Мок-курсы валют (в реальном приложении нужно получать из API)
        # Курсы указаны как количество рублей за 1 единицу валюты
        exchange_rates = {
            "USD": 95.50,
            "EUR": 103.20,
            "GBP": 120.80,
            "CNY": 13.20,
            "JPY": 0.65,
            "RUB": 1.0
        }
        
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        if from_currency not in exchange_rates or to_currency not in exchange_rates:
            return json.dumps({
                "error": f"Неподдерживаемая валюта. Доступны: {', '.join(exchange_rates.keys())}"
            }, ensure_ascii=False)
        
        # Конвертируем через рубли
        # Сначала в рубли
        amount_in_rub = amount * exchange_rates[from_currency]
        # Затем в целевую валюту
        result_amount = amount_in_rub / exchange_rates[to_currency]
        
        # Курс для отображения
        if from_currency == "RUB":
            rate = 1 / exchange_rates[to_currency]
            rate_display = f"1 {to_currency} = {exchange_rates[to_currency]:.2f} RUB"
        elif to_currency == "RUB":
            rate = exchange_rates[from_currency]
            rate_display = f"1 {from_currency} = {exchange_rates[from_currency]:.2f} RUB"
        else:
            # Конвертация между двумя валютами через рубли
            rate = exchange_rates[from_currency] / exchange_rates[to_currency]
            rate_display = f"1 {from_currency} = {rate:.4f} {to_currency}"
        
        result = {
            "amount": round(amount, 2),
            "from_currency": from_currency,
            "to_currency": to_currency,
            "result_amount": round(result_amount, 2),
            "exchange_rate": round(rate, 4),
            "rate_display": rate_display,
            "note": "Курсы валют являются примерными. Для актуальных курсов обратитесь в отделение банка."
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in convert_currency: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def calculate_early_repayment(
    current_balance: float,
    monthly_payment: float,
    interest_rate: float,
    remaining_months: int,
    early_payment_amount: float
) -> str:
    """
    Рассчитывает экономию и новый график платежей при досрочном погашении кредита.
    
    Args:
        current_balance: Текущий остаток долга
        monthly_payment: Текущий ежемесячный платеж
        interest_rate: Процентная ставка годовых
        remaining_months: Оставшийся срок кредита в месяцах
        early_payment_amount: Сумма досрочного погашения
    
    Returns:
        JSON с расчетом: new_balance, new_monthly_payment, savings, new_term_months
    """
    try:
        monthly_rate = interest_rate / 100 / 12
        
        # Новый остаток после досрочного погашения
        new_balance = current_balance - early_payment_amount
        
        if new_balance <= 0:
            return json.dumps({
                "error": "Сумма досрочного погашения превышает остаток долга"
            }, ensure_ascii=False)
        
        # Рассчитываем новый платеж при том же сроке
        if monthly_rate == 0:
            new_monthly_payment = new_balance / remaining_months
        else:
            new_monthly_payment = new_balance * (monthly_rate * (1 + monthly_rate) ** remaining_months) / \
                                ((1 + monthly_rate) ** remaining_months - 1)
        
        # Рассчитываем новый срок при том же платеже
        if monthly_rate == 0:
            new_term_months = int(new_balance / monthly_payment)
        else:
            # Решаем уравнение для нового срока
            if new_balance * monthly_rate >= monthly_payment:
                new_term_months = remaining_months
            else:
                new_term_months = -math.log(1 - (new_balance * monthly_rate) / monthly_payment) / \
                                math.log(1 + monthly_rate)
                new_term_months = math.ceil(new_term_months)
        
        # Старая переплата
        old_total = monthly_payment * remaining_months
        old_overpayment = old_total - current_balance
        
        # Новая переплата
        new_total = new_monthly_payment * remaining_months
        new_overpayment = new_total - new_balance
        
        # Экономия
        savings = old_overpayment - new_overpayment
        
        result = {
            "current_balance": round(current_balance, 2),
            "early_payment_amount": round(early_payment_amount, 2),
            "new_balance": round(new_balance, 2),
            "old_monthly_payment": round(monthly_payment, 2),
            "new_monthly_payment": round(new_monthly_payment, 2),
            "payment_reduction": round(monthly_payment - new_monthly_payment, 2),
            "old_term_months": remaining_months,
            "new_term_months": new_term_months,
            "term_reduction": remaining_months - new_term_months,
            "savings": round(savings, 2),
            "old_overpayment": round(old_overpayment, 2),
            "new_overpayment": round(new_overpayment, 2)
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in calculate_early_repayment: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

