/**
 * Export Utilities - FASE 7
 * Funções para exportar dados em diferentes formatos (CSV, Excel, PDF)
 */

import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

/**
 * Export data to CSV format
 * @param data Array of objects to export
 * @param filename Output filename (without extension)
 */
export const exportToCSV = <T extends Record<string, any>>(
  data: T[],
  filename: string = 'export'
): void => {
  try {
    if (!data || data.length === 0) {
      throw new Error('No data to export');
    }

    // Convert to CSV using papaparse
    const csv = Papa.unparse(data, {
      quotes: true,
      delimiter: ',',
      header: true,
    });

    // Create blob and download
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `${filename}.csv`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error exporting to CSV:', error);
    throw error;
  }
};

/**
 * Export data to Excel format
 * @param data Array of objects to export
 * @param filename Output filename (without extension)
 * @param sheetName Name of the worksheet
 */
export const exportToExcel = <T extends Record<string, any>>(
  data: T[],
  filename: string = 'export',
  sheetName: string = 'Sheet1'
): void => {
  try {
    if (!data || data.length === 0) {
      throw new Error('No data to export');
    }

    // Create workbook and worksheet
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);

    // Auto-size columns
    const maxWidth = 50;
    const colWidths: number[] = [];

    // Get column widths based on content
    const range = XLSX.utils.decode_range(worksheet['!ref'] || 'A1');
    for (let C = range.s.c; C <= range.e.c; ++C) {
      let maxLen = 10; // minimum width

      for (let R = range.s.r; R <= range.e.r; ++R) {
        const cellAddress = XLSX.utils.encode_cell({ r: R, c: C });
        const cell = worksheet[cellAddress];

        if (cell && cell.v) {
          const cellValue = cell.v.toString();
          maxLen = Math.max(maxLen, cellValue.length);
        }
      }

      colWidths.push(Math.min(maxLen + 2, maxWidth));
    }

    worksheet['!cols'] = colWidths.map(w => ({ wch: w }));

    // Generate Excel file
    XLSX.writeFile(workbook, `${filename}.xlsx`);
  } catch (error) {
    console.error('Error exporting to Excel:', error);
    throw error;
  }
};

/**
 * Export HTML element to PDF
 * @param elementId ID of the HTML element to capture
 * @param filename Output filename (without extension)
 * @param options PDF generation options
 */
export const exportToPDF = async (
  elementId: string,
  filename: string = 'export',
  options: {
    orientation?: 'portrait' | 'landscape';
    format?: 'a4' | 'letter';
    quality?: number;
  } = {}
): Promise<void> => {
  try {
    const element = document.getElementById(elementId);

    if (!element) {
      throw new Error(`Element with id "${elementId}" not found`);
    }

    const {
      orientation = 'portrait',
      format = 'a4',
      quality = 0.95,
    } = options;

    // Capture element as canvas
    const canvas = await html2canvas(element, {
      scale: 2, // Higher quality
      useCORS: true,
      logging: false,
      windowWidth: element.scrollWidth,
      windowHeight: element.scrollHeight,
    });

    // Convert canvas to image
    const imgData = canvas.toDataURL('image/png', quality);

    // Calculate PDF dimensions
    const pdf = new jsPDF({
      orientation,
      unit: 'mm',
      format,
    });

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();

    const imgWidth = canvas.width;
    const imgHeight = canvas.height;

    // Calculate scaling to fit page
    const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
    const scaledWidth = imgWidth * ratio;
    const scaledHeight = imgHeight * ratio;

    // Center image on page
    const x = (pdfWidth - scaledWidth) / 2;
    const y = (pdfHeight - scaledHeight) / 2;

    pdf.addImage(imgData, 'PNG', x, y, scaledWidth, scaledHeight);
    pdf.save(`${filename}.pdf`);
  } catch (error) {
    console.error('Error exporting to PDF:', error);
    throw error;
  }
};

/**
 * Export data to PDF as table
 * @param data Array of objects to export
 * @param filename Output filename (without extension)
 * @param title Document title
 * @param columns Column configuration
 */
export const exportDataToPDF = <T extends Record<string, any>>(
  data: T[],
  filename: string = 'export',
  title: string = 'Data Export',
  columns?: { header: string; dataKey: string }[]
): void => {
  try {
    if (!data || data.length === 0) {
      throw new Error('No data to export');
    }

    const pdf = new jsPDF('landscape', 'mm', 'a4');
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();

    // Add title
    pdf.setFontSize(16);
    pdf.setFont('helvetica', 'bold');
    pdf.text(title, pageWidth / 2, 15, { align: 'center' });

    // Add timestamp
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    const timestamp = new Date().toLocaleString('pt-BR');
    pdf.text(`Generated: ${timestamp}`, pageWidth / 2, 22, { align: 'center' });

    // Prepare table data
    const headers = columns
      ? columns.map(col => col.header)
      : Object.keys(data[0]);

    const dataKeys = columns
      ? columns.map(col => col.dataKey)
      : Object.keys(data[0]);

    const rows = data.map(row =>
      dataKeys.map(key => {
        const value = row[key];
        if (value === null || value === undefined) return '-';
        if (typeof value === 'object') return JSON.stringify(value);
        return value.toString();
      })
    );

    // Simple table rendering (manual implementation since autoTable might not be available)
    let startY = 30;
    const rowHeight = 8;
    const colWidth = (pageWidth - 20) / headers.length;

    // Draw headers
    pdf.setFillColor(51, 122, 183); // Primary blue
    pdf.setTextColor(255, 255, 255);
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'bold');

    headers.forEach((header, i) => {
      pdf.rect(10 + i * colWidth, startY, colWidth, rowHeight, 'F');
      pdf.text(header, 12 + i * colWidth, startY + 5.5);
    });

    // Draw data rows
    pdf.setTextColor(0, 0, 0);
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(9);

    rows.forEach((row, rowIndex) => {
      const y = startY + (rowIndex + 1) * rowHeight;

      // Check if we need a new page
      if (y + rowHeight > pageHeight - 20) {
        pdf.addPage();
        startY = 15;
        return;
      }

      // Alternate row colors
      if (rowIndex % 2 === 0) {
        pdf.setFillColor(245, 245, 245);
        pdf.rect(10, y, pageWidth - 20, rowHeight, 'F');
      }

      row.forEach((cell, cellIndex) => {
        // Truncate long text
        let text = cell.length > 30 ? cell.substring(0, 27) + '...' : cell;
        pdf.text(text, 12 + cellIndex * colWidth, y + 5.5);
      });
    });

    // Add footer
    const totalPages = (pdf as any).internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      pdf.setPage(i);
      pdf.setFontSize(8);
      pdf.setTextColor(128, 128, 128);
      pdf.text(
        `Page ${i} of ${totalPages}`,
        pageWidth / 2,
        pageHeight - 10,
        { align: 'center' }
      );
    }

    pdf.save(`${filename}.pdf`);
  } catch (error) {
    console.error('Error exporting data to PDF:', error);
    throw error;
  }
};

/**
 * Download JSON data as file
 * @param data Data to export
 * @param filename Output filename (without extension)
 */
export const exportToJSON = <T>(
  data: T,
  filename: string = 'export'
): void => {
  try {
    const jsonString = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `${filename}.json`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error exporting to JSON:', error);
    throw error;
  }
};

/**
 * Export interface for React components
 */
export const exportUtils = {
  toCSV: exportToCSV,
  toExcel: exportToExcel,
  toPDF: exportToPDF,
  dataToPDF: exportDataToPDF,
  toJSON: exportToJSON,
};

export default exportUtils;
