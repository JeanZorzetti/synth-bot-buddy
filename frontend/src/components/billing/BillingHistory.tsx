/**
 * 📄 BILLING HISTORY & INVOICE MANAGEMENT
 * Complete billing history with invoice downloads and payment tracking
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { apiClient } from '../../services/apiClient';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  Download,
  Search,
  Filter,
  Calendar,
  CreditCard,
  CheckCircle,
  AlertTriangle,
  Clock,
  X,
  RefreshCw,
  FileText,
  DollarSign,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Eye,
  Mail,
  Printer,
  Share2,
  Archive,
  Star,
  AlertCircle
} from 'lucide-react';

interface Invoice {
  id: string;
  number: string;
  date: string;
  dueDate: string;
  amount: number;
  currency: string;
  status: 'paid' | 'pending' | 'overdue' | 'cancelled' | 'refunded';
  description: string;
  items: InvoiceItem[];
  paymentMethod?: string;
  paidAt?: string;
  downloadUrl?: string;
  subscription?: {
    plan: string;
    period: string;
  };
  taxes?: {
    amount: number;
    rate: number;
    type: string;
  }[];
  discounts?: {
    amount: number;
    code?: string;
    description: string;
  }[];
  customer: {
    name: string;
    email: string;
    address?: {
      line1: string;
      city: string;
      state: string;
      country: string;
      postalCode: string;
    };
  };
}

interface InvoiceItem {
  id: string;
  description: string;
  quantity: number;
  unitPrice: number;
  amount: number;
  period?: {
    start: string;
    end: string;
  };
}

interface PaymentTransaction {
  id: string;
  invoiceId: string;
  date: string;
  amount: number;
  currency: string;
  status: 'successful' | 'failed' | 'pending' | 'cancelled' | 'refunded';
  paymentMethod: string;
  gateway: string;
  gatewayTransactionId?: string;
  failureReason?: string;
  refundReason?: string;
  fees?: {
    amount: number;
    type: string;
  }[];
}

interface BillingStats {
  totalSpent: number;
  averageMonthly: number;
  currentMonth: number;
  pendingAmount: number;
  currency: string;
  period: {
    start: string;
    end: string;
  };
}

interface BillingHistoryProps {
  className?: string;
}

export function BillingHistory({ className = "" }: BillingHistoryProps) {
  const { user } = useAuth();

  const [activeTab, setActiveTab] = useState('invoices');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [transactions, setTransactions] = useState<PaymentTransaction[]>([]);
  const [billingStats, setBillingStats] = useState<BillingStats | null>(null);

  // Filters and search
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [dateFilter, setDateFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'date' | 'amount' | 'status'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // UI states
  const [downloadingInvoice, setDownloadingInvoice] = useState<string | null>(null);
  const [selectedInvoice, setSelectedInvoice] = useState<Invoice | null>(null);

  useEffect(() => {
    loadBillingData();
  }, []);

  const loadBillingData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [invoicesData, transactionsData, statsData] = await Promise.all([
        apiClient.get<Invoice[]>('/billing/invoices'),
        apiClient.get<PaymentTransaction[]>('/billing/transactions'),
        apiClient.get<BillingStats>('/billing/stats')
      ]);

      setInvoices(invoicesData);
      setTransactions(transactionsData);
      setBillingStats(statsData);
    } catch (error: any) {
      setError(error.message || 'Failed to load billing data');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadInvoice = async (invoiceId: string, format: 'pdf' | 'csv' = 'pdf') => {
    try {
      setDownloadingInvoice(invoiceId);

      const response = await apiClient.get(`/billing/invoices/${invoiceId}/download?format=${format}`, {
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response]));
      const link = document.createElement('a');
      link.href = url;
      link.download = `invoice-${invoiceId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

    } catch (error: any) {
      setError(error.message || 'Failed to download invoice');
    } finally {
      setDownloadingInvoice(null);
    }
  };

  const sendInvoiceByEmail = async (invoiceId: string) => {
    try {
      await apiClient.post(`/billing/invoices/${invoiceId}/send-email`);
      alert('Invoice sent to your email successfully!');
    } catch (error: any) {
      setError(error.message || 'Failed to send invoice');
    }
  };

  const requestRefund = async (transactionId: string, reason: string) => {
    try {
      await apiClient.post(`/billing/transactions/${transactionId}/refund`, {
        reason
      });
      await loadBillingData();
      alert('Refund request submitted successfully!');
    } catch (error: any) {
      setError(error.message || 'Failed to request refund');
    }
  };

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
    }).format(amount);
  };

  const formatDate = (dateString: string, includeTime: boolean = false) => {
    const options: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    };

    if (includeTime) {
      options.hour = '2-digit';
      options.minute = '2-digit';
    }

    return new Date(dateString).toLocaleDateString('en-US', options);
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      paid: { className: 'bg-green-100 text-green-800', icon: CheckCircle },
      successful: { className: 'bg-green-100 text-green-800', icon: CheckCircle },
      pending: { className: 'bg-yellow-100 text-yellow-800', icon: Clock },
      overdue: { className: 'bg-red-100 text-red-800', icon: AlertTriangle },
      failed: { className: 'bg-red-100 text-red-800', icon: X },
      cancelled: { className: 'bg-gray-100 text-gray-800', icon: X },
      refunded: { className: 'bg-blue-100 text-blue-800', icon: RefreshCw },
    };

    const variant = variants[status as keyof typeof variants] || variants.pending;
    const Icon = variant.icon;

    return (
      <Badge className={variant.className}>
        <Icon className="h-3 w-3 mr-1" />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const filterInvoices = () => {
    let filtered = [...invoices];

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(invoice =>
        invoice.number.toLowerCase().includes(searchTerm.toLowerCase()) ||
        invoice.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(invoice => invoice.status === statusFilter);
    }

    // Date filter
    if (dateFilter !== 'all') {
      const now = new Date();
      const filterDate = new Date();

      switch (dateFilter) {
        case 'last30':
          filterDate.setDate(now.getDate() - 30);
          break;
        case 'last90':
          filterDate.setDate(now.getDate() - 90);
          break;
        case 'lastyear':
          filterDate.setFullYear(now.getFullYear() - 1);
          break;
      }

      filtered = filtered.filter(invoice => new Date(invoice.date) >= filterDate);
    }

    // Sort
    filtered.sort((a, b) => {
      let aValue, bValue;

      switch (sortBy) {
        case 'amount':
          aValue = a.amount;
          bValue = b.amount;
          break;
        case 'status':
          aValue = a.status;
          bValue = b.status;
          break;
        default:
          aValue = new Date(a.date).getTime();
          bValue = new Date(b.date).getTime();
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  };

  const filterTransactions = () => {
    let filtered = [...transactions];

    if (searchTerm) {
      filtered = filtered.filter(transaction =>
        transaction.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        transaction.paymentMethod.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(transaction => transaction.status === statusFilter);
    }

    return filtered.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  };

  const renderBillingStats = () => (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <DollarSign className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {billingStats ? formatCurrency(billingStats.totalSpent, billingStats.currency) : '$0'}
              </p>
              <p className="text-sm text-gray-500">Total Spent</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <TrendingUp className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {billingStats ? formatCurrency(billingStats.averageMonthly, billingStats.currency) : '$0'}
              </p>
              <p className="text-sm text-gray-500">Avg Monthly</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <BarChart3 className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {billingStats ? formatCurrency(billingStats.currentMonth, billingStats.currency) : '$0'}
              </p>
              <p className="text-sm text-gray-500">This Month</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-orange-100 rounded-lg">
              <Clock className="h-5 w-5 text-orange-600" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {billingStats ? formatCurrency(billingStats.pendingAmount, billingStats.currency) : '$0'}
              </p>
              <p className="text-sm text-gray-500">Pending</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderFilters = () => (
    <Card className="mb-6">
      <CardContent className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search invoices..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 border rounded-md bg-white"
          >
            <option value="all">All Status</option>
            <option value="paid">Paid</option>
            <option value="pending">Pending</option>
            <option value="overdue">Overdue</option>
            <option value="cancelled">Cancelled</option>
            <option value="refunded">Refunded</option>
          </select>

          {/* Date Filter */}
          <select
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            className="px-3 py-2 border rounded-md bg-white"
          >
            <option value="all">All Time</option>
            <option value="last30">Last 30 Days</option>
            <option value="last90">Last 90 Days</option>
            <option value="lastyear">Last Year</option>
          </select>

          {/* Sort Options */}
          <div className="flex space-x-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-2 border rounded-md bg-white flex-1"
            >
              <option value="date">Sort by Date</option>
              <option value="amount">Sort by Amount</option>
              <option value="status">Sort by Status</option>
            </select>

            <Button
              variant="outline"
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="px-3"
            >
              {sortOrder === 'asc' ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        <div className="flex justify-between items-center mt-4">
          <p className="text-sm text-gray-500">
            Showing {activeTab === 'invoices' ? filterInvoices().length : filterTransactions().length} results
          </p>

          <Button
            variant="outline"
            onClick={loadBillingData}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderInvoices = () => (
    <div className="space-y-4">
      {filterInvoices().map((invoice) => (
        <Card key={invoice.id} className="hover:shadow-md transition-shadow">
          <CardContent className="p-6">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="font-semibold">Invoice #{invoice.number}</h3>
                  {getStatusBadge(invoice.status)}
                  {invoice.subscription && (
                    <Badge variant="outline">
                      {invoice.subscription.plan} - {invoice.subscription.period}
                    </Badge>
                  )}
                </div>

                <p className="text-gray-600 mb-2">{invoice.description}</p>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500">Date</p>
                    <p className="font-medium">{formatDate(invoice.date)}</p>
                  </div>

                  <div>
                    <p className="text-gray-500">Due Date</p>
                    <p className="font-medium">{formatDate(invoice.dueDate)}</p>
                  </div>

                  <div>
                    <p className="text-gray-500">Amount</p>
                    <p className="font-medium text-lg">
                      {formatCurrency(invoice.amount, invoice.currency)}
                    </p>
                  </div>

                  <div>
                    <p className="text-gray-500">Payment Method</p>
                    <p className="font-medium">{invoice.paymentMethod || 'Not specified'}</p>
                  </div>
                </div>

                {invoice.taxes && invoice.taxes.length > 0 && (
                  <div className="mt-3 text-sm">
                    <p className="text-gray-500">Taxes:</p>
                    {invoice.taxes.map((tax, index) => (
                      <p key={index} className="text-gray-600">
                        {tax.type}: {formatCurrency(tax.amount, invoice.currency)} ({tax.rate}%)
                      </p>
                    ))}
                  </div>
                )}

                {invoice.discounts && invoice.discounts.length > 0 && (
                  <div className="mt-3 text-sm">
                    <p className="text-gray-500">Discounts:</p>
                    {invoice.discounts.map((discount, index) => (
                      <p key={index} className="text-green-600">
                        {discount.description}: -{formatCurrency(discount.amount, invoice.currency)}
                        {discount.code && ` (${discount.code})`}
                      </p>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex flex-col space-y-2 ml-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedInvoice(invoice)}
                >
                  <Eye className="h-4 w-4 mr-2" />
                  View
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => downloadInvoice(invoice.id, 'pdf')}
                  disabled={downloadingInvoice === invoice.id}
                >
                  {downloadingInvoice === invoice.id ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4 mr-2" />
                  )}
                  PDF
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => sendInvoiceByEmail(invoice.id)}
                >
                  <Mail className="h-4 w-4 mr-2" />
                  Email
                </Button>

                {invoice.status === 'paid' && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const reason = prompt('Please provide a reason for the refund request:');
                      if (reason) {
                        // Find the transaction for this invoice
                        const transaction = transactions.find(t => t.invoiceId === invoice.id);
                        if (transaction) {
                          requestRefund(transaction.id, reason);
                        }
                      }
                    }}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refund
                  </Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}

      {filterInvoices().length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No invoices found</p>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderTransactions = () => (
    <div className="space-y-4">
      {filterTransactions().map((transaction) => (
        <Card key={transaction.id} className="hover:shadow-md transition-shadow">
          <CardContent className="p-6">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="font-semibold">Transaction #{transaction.id.slice(-8)}</h3>
                  {getStatusBadge(transaction.status)}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500">Date</p>
                    <p className="font-medium">{formatDate(transaction.date, true)}</p>
                  </div>

                  <div>
                    <p className="text-gray-500">Amount</p>
                    <p className="font-medium text-lg">
                      {formatCurrency(transaction.amount, transaction.currency)}
                    </p>
                  </div>

                  <div>
                    <p className="text-gray-500">Payment Method</p>
                    <p className="font-medium">{transaction.paymentMethod}</p>
                  </div>

                  <div>
                    <p className="text-gray-500">Gateway</p>
                    <p className="font-medium">{transaction.gateway}</p>
                  </div>
                </div>

                {transaction.gatewayTransactionId && (
                  <div className="mt-3 text-sm">
                    <p className="text-gray-500">Gateway Transaction ID:</p>
                    <p className="font-mono text-gray-600">{transaction.gatewayTransactionId}</p>
                  </div>
                )}

                {transaction.failureReason && (
                  <div className="mt-3 text-sm">
                    <p className="text-red-500">Failure Reason:</p>
                    <p className="text-red-600">{transaction.failureReason}</p>
                  </div>
                )}

                {transaction.refundReason && (
                  <div className="mt-3 text-sm">
                    <p className="text-blue-500">Refund Reason:</p>
                    <p className="text-blue-600">{transaction.refundReason}</p>
                  </div>
                )}

                {transaction.fees && transaction.fees.length > 0 && (
                  <div className="mt-3 text-sm">
                    <p className="text-gray-500">Fees:</p>
                    {transaction.fees.map((fee, index) => (
                      <p key={index} className="text-gray-600">
                        {fee.type}: {formatCurrency(fee.amount, transaction.currency)}
                      </p>
                    ))}
                  </div>
                )}
              </div>

              <div className="ml-4">
                <div className={`p-3 rounded-lg ${
                  transaction.status === 'successful' ? 'bg-green-100' :
                  transaction.status === 'failed' ? 'bg-red-100' :
                  transaction.status === 'pending' ? 'bg-yellow-100' :
                  'bg-gray-100'
                }`}>
                  <CreditCard className={`h-6 w-6 ${
                    transaction.status === 'successful' ? 'text-green-600' :
                    transaction.status === 'failed' ? 'text-red-600' :
                    transaction.status === 'pending' ? 'text-yellow-600' :
                    'text-gray-600'
                  }`} />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}

      {filterTransactions().length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <CreditCard className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No transactions found</p>
          </CardContent>
        </Card>
      )}
    </div>
  );

  if (!user) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Please log in to view your billing history.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`max-w-6xl mx-auto space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Billing History</h1>
          <p className="text-gray-600">View your invoices, payments, and transaction history</p>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {isLoading ? (
        <div className="space-y-4">
          {[1, 2, 3, 4].map(i => (
            <Card key={i}>
              <CardContent className="p-6">
                <div className="animate-pulse space-y-4">
                  <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                  <div className="h-8 bg-gray-200 rounded w-1/2"></div>
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <>
          {/* Billing Stats */}
          {billingStats && renderBillingStats()}

          {/* Filters */}
          {renderFilters()}

          {/* Main Content */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2 max-w-md">
              <TabsTrigger value="invoices" className="flex items-center space-x-2">
                <FileText className="h-4 w-4" />
                <span>Invoices</span>
              </TabsTrigger>
              <TabsTrigger value="transactions" className="flex items-center space-x-2">
                <CreditCard className="h-4 w-4" />
                <span>Transactions</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="invoices">
              {renderInvoices()}
            </TabsContent>

            <TabsContent value="transactions">
              {renderTransactions()}
            </TabsContent>
          </Tabs>
        </>
      )}

      {/* Invoice Detail Modal */}
      {selectedInvoice && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Invoice #{selectedInvoice.number}</CardTitle>
                <Button
                  variant="ghost"
                  onClick={() => setSelectedInvoice(null)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              {/* Invoice Details */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Invoice Date</p>
                  <p className="font-medium">{formatDate(selectedInvoice.date)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Due Date</p>
                  <p className="font-medium">{formatDate(selectedInvoice.dueDate)}</p>
                </div>
              </div>

              {/* Customer Info */}
              <div>
                <h4 className="font-medium mb-2">Bill To:</h4>
                <p>{selectedInvoice.customer.name}</p>
                <p>{selectedInvoice.customer.email}</p>
                {selectedInvoice.customer.address && (
                  <div className="text-sm text-gray-600">
                    <p>{selectedInvoice.customer.address.line1}</p>
                    <p>
                      {selectedInvoice.customer.address.city}, {selectedInvoice.customer.address.state} {selectedInvoice.customer.address.postalCode}
                    </p>
                    <p>{selectedInvoice.customer.address.country}</p>
                  </div>
                )}
              </div>

              {/* Invoice Items */}
              <div>
                <h4 className="font-medium mb-2">Items:</h4>
                <div className="space-y-2">
                  {selectedInvoice.items.map((item) => (
                    <div key={item.id} className="flex justify-between items-center p-3 border rounded">
                      <div>
                        <p className="font-medium">{item.description}</p>
                        {item.period && (
                          <p className="text-sm text-gray-500">
                            {formatDate(item.period.start)} - {formatDate(item.period.end)}
                          </p>
                        )}
                        <p className="text-sm">
                          {item.quantity} × {formatCurrency(item.unitPrice, selectedInvoice.currency)}
                        </p>
                      </div>
                      <p className="font-medium">
                        {formatCurrency(item.amount, selectedInvoice.currency)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Total */}
              <div className="border-t pt-4">
                <div className="flex justify-between items-center text-lg font-bold">
                  <span>Total</span>
                  <span>{formatCurrency(selectedInvoice.amount, selectedInvoice.currency)}</span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex space-x-2 pt-4">
                <Button
                  onClick={() => downloadInvoice(selectedInvoice.id, 'pdf')}
                  disabled={downloadingInvoice === selectedInvoice.id}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download PDF
                </Button>
                <Button
                  variant="outline"
                  onClick={() => sendInvoiceByEmail(selectedInvoice.id)}
                >
                  <Mail className="h-4 w-4 mr-2" />
                  Email Invoice
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}