/**
 * Trading Page
 * Página simplificada para execução de ordens - Objetivo 1
 */

import { OrderExecutor } from '@/components/orders/OrderExecutor';

export default function Trading() {
  return (
    <div className="container mx-auto p-6">
      <OrderExecutor />
    </div>
  );
}
